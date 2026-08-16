import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import PlainTextResponse, RedirectResponse

from app.api.auth import ApiKeyContext, record_usage, require_api_key
from app.mcp.server import create_mcp_http_app, create_mcp_server
from app.rag.config import Settings, get_settings
from app.rag.ingest import control_state_lock, migrate_legacy_control_state
from app.rag.models import (
    ImageContextResponse,
    PageContextResponse,
    PageTextResponse,
    SearchRequest,
    SearchResponse,
    SimilarPagesResponse,
    SourceListResponse,
    SourceSearchRequest,
    SourceSearchResponse,
    SourceSummary,
)
from app.rag.search import search
from app.rag.store import RagStore
from app.rag.voyage_client import MissingVoyageApiKey

settings = get_settings()
LOGGER = logging.getLogger(__name__)
mcp_server = create_mcp_server(settings)
mcp_http_app = create_mcp_http_app(mcp_server, settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await prepare_control_state_async(settings)
    await _ensure_payload_indexes()
    async with mcp_server.session_manager.run():
        yield


def prepare_control_state(current_settings: Settings) -> None:
    with control_state_lock(current_settings, blocking=True):
        migrate_legacy_control_state(current_settings)


async def prepare_control_state_async(current_settings: Settings) -> None:
    await _run_blocking_safely(partial(prepare_control_state, current_settings))


async def _ensure_payload_indexes() -> None:
    """Create declared Qdrant indexes before accepting traffic."""
    store = RagStore(settings)
    try:
        await _run_blocking_safely(store.ensure_payload_indexes)
    except Exception:
        LOGGER.warning("Could not ensure Qdrant payload indexes during startup.", exc_info=True)
    finally:
        try:
            store.client.close()
        except Exception:
            LOGGER.warning("Could not close the startup Qdrant client.", exc_info=True)


async def _run_blocking_safely(operation: Callable[[], object]) -> None:
    worker = asyncio.create_task(asyncio.to_thread(operation))
    cancellation: asyncio.CancelledError | None = None
    worker_error: BaseException | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
        except BaseException as exc:
            worker_error = exc
            break
    if cancellation is not None:
        if not worker.cancelled():
            worker.exception()
        raise cancellation
    if worker_error is not None:
        raise worker_error
    worker.result()


PRIVATE_ARTIFACT_ROOTS = frozenset(
    {"active-generations.json", "ingestion-manifest.jsonl", "ingestion.lock"}
)


class ArtifactStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        path_parts = path.split("/")
        if any(part.startswith(".") or part in PRIVATE_ARTIFACT_ROOTS for part in path_parts):
            return PlainTextResponse("Not Found", status_code=404)
        return await super().get_response(path, scope)


app = FastAPI(title="FRC Mechanism RAG", version="0.1.0", lifespan=lifespan)
app.mount(
    settings.artifact_url_base,
    ArtifactStaticFiles(directory=settings.artifact_dir, check_dir=False),
    name="images",
)
if settings.artifact_url_base != "/artifacts":
    app.mount(
        "/artifacts",
        ArtifactStaticFiles(directory=settings.artifact_dir, check_dir=False),
        name="artifacts",
    )


@app.middleware("http")
async def usage_recording_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    api_key_context = getattr(request.state, "api_key_context", None)
    if isinstance(api_key_context, ApiKeyContext):
        record_usage(
            context=api_key_context,
            request=request,
            status_code=response.status_code,
            started_at=started_at,
        )
    return response


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"ok": True, "collection": settings.collection_name}


@app.get("/auth/validate")
def validate_api_key(api_key: ApiKeyContext = Depends(require_api_key)) -> dict:
    return {
        "valid": True,
        "apiKeyId": api_key.api_key_id,
        "workspaceId": api_key.organization_id,
        "permissions": list(api_key.permissions),
    }


@app.post("/search", response_model=SearchResponse)
def search_endpoint(
    request: SearchRequest,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> SearchResponse:
    try:
        return search(
            request.query,
            top_k=request.top_k,
            debug=request.debug,
            settings=get_settings(),
            team=request.team,
            year=request.year,
            source=request.source,
            team_numbers=request.team_numbers,
            years=request.years,
            source_ids=request.source_ids,
            mechanism_types=request.mechanism_types,
            sort=request.sort,
            modality=request.modality,
        )
    except MissingVoyageApiKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/sources", response_model=SourceListResponse)
def list_sources(
    team: str | None = None,
    year: int | None = None,
    source: str | None = None,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> SourceListResponse:
    return RagStore(get_settings()).list_sources(team=team, year=year, source=source)


@app.post("/sources/search", response_model=SourceSearchResponse)
def search_sources(
    request: SourceSearchRequest,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> SourceSearchResponse:
    if request.query:
        search_response = search(
            request.query,
            top_k=max(request.top_k * 3, request.top_k),
            debug=False,
            settings=get_settings(),
            team=request.team,
            year=request.year,
            source=request.source,
        )
        matches = RagStore(get_settings()).source_search_from_results(
            request.query, search_response.results
        )
        return SourceSearchResponse(query=request.query, matches=matches[: request.top_k])

    sources = RagStore(get_settings()).list_sources(
        team=request.team,
        year=request.year,
        source=request.source,
    )
    source_matches = []
    for source_item in sources.sources[: request.top_k]:
        page = source_item.pages[0] if source_item.pages else 0
        context_url, text_url = _source_page_urls(source_item, page)
        source_matches.append(
            {
                "source_version_id": source_item.source_version_id,
                "ingestion_id": source_item.ingestion_id,
                "source_pdf": source_item.source_pdf,
                "team": source_item.team,
                "year": source_item.year,
                "page": page,
                "score": 1.0,
                "best_snippets": [],
                "image_urls": source_item.sample_image_urls,
                "page_context_url": context_url,
                "page_text_url": text_url,
            }
        )
    return SourceSearchResponse(query=None, matches=source_matches)


@app.get("/sources/{source_pdf}", response_model=SourceSummary)
def source_summary(
    source_pdf: str,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> SourceSummary:
    summary = RagStore(get_settings()).source_summary(source_pdf)
    if summary is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return summary


@app.get("/similar", response_model=SimilarPagesResponse)
def similar_pages(
    result_id: str | None = None,
    source_pdf: str | None = None,
    page: int | None = None,
    top_k: int = Query(default=10, ge=1, le=20),
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> SimilarPagesResponse:
    store = RagStore(get_settings())
    if result_id:
        response = store.similar_from_result_id(result_id, top_k)
    elif source_pdf and page is not None:
        response = store.similar_from_page(source_pdf, page, top_k)
    else:
        raise HTTPException(status_code=400, detail="Provide result_id or source_pdf + page.")
    if response is None:
        raise HTTPException(status_code=404, detail="Similar page seed not found.")
    return response


@app.get("/image-context", response_model=ImageContextResponse)
def image_context(
    result_id: str | None = None,
    image_url: str | None = None,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> ImageContextResponse:
    if not result_id and not image_url:
        raise HTTPException(status_code=400, detail="Provide result_id or image_url.")
    context = RagStore(get_settings()).image_context(result_id=result_id, image_url=image_url)
    if context is None:
        raise HTTPException(status_code=404, detail="Image context not found.")
    return context


@app.get("/pages/{source_pdf}/{page}", response_model=PageContextResponse)
def page_context(
    source_pdf: str,
    page: int,
    source_version_id: str | None = None,
    ingestion_id: str | None = None,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> PageContextResponse:
    context = RagStore(get_settings()).page_context(
        source_pdf, page, source_version_id, ingestion_id
    )
    if context is None:
        raise HTTPException(status_code=404, detail="Page context not found.")
    return context


@app.get("/pages/{source_pdf}/{page}/text", response_model=PageTextResponse)
def page_text(
    source_pdf: str,
    page: int,
    source_version_id: str | None = None,
    ingestion_id: str | None = None,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> PageTextResponse:
    context = RagStore(get_settings()).page_context(
        source_pdf, page, source_version_id, ingestion_id
    )
    if context is None or not context.text.strip():
        raise HTTPException(status_code=404, detail="Page text not found.")
    return PageTextResponse(source_pdf=source_pdf, page=page, text=context.text)


@app.post("/collections/init")
def init_collection(_api_key: ApiKeyContext = Depends(require_api_key)) -> dict:
    RagStore(get_settings()).ensure_collection()
    return {"ok": True}


def _source_page_urls(source: SourceSummary, page: int) -> tuple[str, str]:
    if not page:
        return "", ""
    path = f"/pages/{quote(source.source_pdf, safe='')}/{page}"
    return path, f"{path}/text"


@app.get("/.well-known/oauth-authorization-server", include_in_schema=False)
def legacy_oauth_metadata() -> RedirectResponse:
    """Support clients that still discover OAuth metadata from the resource origin."""
    issuer = str(settings.clerk_oauth_issuer_url).rstrip("/")
    return RedirectResponse(
        f"{issuer}/.well-known/oauth-authorization-server",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# This catch-all mount must remain last so the existing REST and artifact routes win.
app.mount("/", mcp_http_app, name="mcp")
