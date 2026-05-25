import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote

from app.api.auth import ApiKeyContext, record_usage, require_api_key
from app.rag.config import get_settings
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

app = FastAPI(title="FRC Mechanism RAG", version="0.1.0")
settings = get_settings()
app.mount(
    settings.artifact_url_base,
    StaticFiles(directory=settings.artifact_dir, check_dir=False),
    name="images",
)
if settings.artifact_url_base != "/artifacts":
    app.mount(
        "/artifacts",
        StaticFiles(directory=settings.artifact_dir, check_dir=False),
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
        matches = RagStore(get_settings()).source_search_from_results(request.query, search_response.results)
        return SourceSearchResponse(query=request.query, matches=matches[: request.top_k])

    sources = RagStore(get_settings()).list_sources(
        team=request.team,
        year=request.year,
        source=request.source,
    )
    source_matches = []
    for source_item in sources.sources[: request.top_k]:
        page = source_item.pages[0] if source_item.pages else 0
        source_matches.append(
            {
                "source_pdf": source_item.source_pdf,
                "team": source_item.team,
                "year": source_item.year,
                "page": page,
                "score": 1.0,
                "best_snippets": [],
                "image_urls": source_item.sample_image_urls,
                "page_context_url": f"/pages/{quote(source_item.source_pdf, safe='')}/{page}" if page else "",
                "page_text_url": f"/pages/{quote(source_item.source_pdf, safe='')}/{page}/text" if page else "",
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
    top_k: int = 10,
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
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> PageContextResponse:
    context = RagStore(get_settings()).page_context(source_pdf, page)
    if context is None:
        raise HTTPException(status_code=404, detail="Page context not found.")
    return context


@app.get("/pages/{source_pdf}/{page}/text", response_model=PageTextResponse)
def page_text(
    source_pdf: str,
    page: int,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> PageTextResponse:
    context = RagStore(get_settings()).page_context(source_pdf, page)
    if context is None or not context.text.strip():
        raise HTTPException(status_code=404, detail="Page text not found.")
    return PageTextResponse(source_pdf=source_pdf, page=page, text=context.text)


@app.post("/collections/init")
def init_collection(_api_key: ApiKeyContext = Depends(require_api_key)) -> dict:
    RagStore(get_settings()).ensure_collection()
    return {"ok": True}
