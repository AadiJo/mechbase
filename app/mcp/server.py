from __future__ import annotations

import base64
import logging
from typing import Annotated, Protocol
from urllib.parse import urlparse

from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import Annotations, CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import Field
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from app.mcp.auth import ClerkTokenVerifier
from app.mcp.cache import TTLCache
from app.mcp.images import load_preview_image
from app.mcp.results import (
    AppliedSearchFilters,
    BrowseCursor,
    BrowseSourceOutput,
    FetchOutput,
    InspectOutput,
    RenderOutput,
    SearchOutput,
    SimilarOutput,
    SourceOutput,
    browse_source_output,
    fetch_output,
    render_output,
    search_output,
    similar_output,
    source_output,
    visual_candidate,
)
from app.mcp.widget import SELECTED_RESULTS_WIDGET_HTML, SELECTED_RESULTS_WIDGET_URI
from app.rag.chunking import contains_token_phrase, normalize_token_phrase
from app.rag.config import Settings
from app.rag.models import (
    FetchContextResponse,
    ImageContextResponse,
    MechanismTypeFilter,
    PageContextResponse,
    SearchRequest,
    SearchResponse,
    SearchSort,
    SimilarPagesResponse,
    SourceBrowseResponse,
    SourceIdFilter,
    SourceListResponse,
    SourceSummary,
)
from app.rag.search import search as rag_search
from app.rag.store import RagStore

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
MODEL_ONLY = Annotations(audience=["assistant"], priority=1.0)
LOGGER = logging.getLogger(__name__)
TeamNumber = Annotated[int, Field(ge=1, le=99999)]
SeasonYear = Annotated[int, Field(ge=1992, le=2100)]
SourceId = SourceIdFilter
MechanismType = MechanismTypeFilter
ResultId = Annotated[str, Field(min_length=1, max_length=256)]
BROWSE_SECTION_SCAN_PAGES = 50


class RetrievalBackend(Protocol):
    def search(self, request: SearchRequest) -> SearchResponse: ...

    def fetch(self, result_id: str) -> ImageContextResponse | None: ...

    def fetch_contexts(
        self,
        result_id: str,
        adjacent_pages: int,
    ) -> FetchContextResponse | None: ...

    def find_similar(
        self,
        result_id: str,
        top_k: int,
        *,
        team_numbers: list[str],
        years: list[int],
        source_ids: list[str],
    ) -> SimilarPagesResponse | None: ...

    def page_contexts(
        self,
        source_pdf: str,
        pages: list[int] | None,
        source_version_id: str | None,
        ingestion_id: str | None,
    ) -> list[PageContextResponse]: ...

    def browse_contexts(
        self,
        source_id: str,
        *,
        start_page: int | None,
        end_page: int | None,
        resume_page: int | None,
        scan_limit: int,
    ) -> SourceBrowseResponse | None: ...

    def list_sources(
        self,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
        source_query: str | None = None,
    ) -> SourceListResponse: ...

    def corpus_revision(self) -> str: ...


class RagRetrievalBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = RagStore(settings)
        self._last_revision = "unknown"

    def search(self, request: SearchRequest) -> SearchResponse:
        return rag_search(
            request.query,
            top_k=request.top_k,
            debug=False,
            settings=self._settings,
            team_numbers=request.team_numbers,
            years=request.years,
            source_ids=request.source_ids,
            mechanism_types=request.mechanism_types,
            sort=request.sort,
            store=self._store,
        )

    def corpus_revision(self) -> str:
        try:
            revision = self._store.corpus_revision()
        except (ApiException, ResponseHandlingException):
            LOGGER.warning("Could not refresh the Qdrant corpus revision; using the last value.")
            return self._last_revision
        self._last_revision = revision
        return revision

    def fetch(self, result_id: str) -> ImageContextResponse | None:
        return self._store.image_context(result_id=result_id)

    def fetch_contexts(
        self,
        result_id: str,
        adjacent_pages: int,
    ) -> FetchContextResponse | None:
        return self._store.fetch_contexts(result_id, adjacent_pages)

    def find_similar(
        self,
        result_id: str,
        top_k: int,
        *,
        team_numbers: list[str],
        years: list[int],
        source_ids: list[str],
    ) -> SimilarPagesResponse | None:
        return self._store.similar_from_result_id(
            result_id,
            top_k,
            team_numbers=team_numbers,
            years=years,
            source_ids=source_ids,
        )

    def page_contexts(
        self,
        source_pdf: str,
        pages: list[int] | None,
        source_version_id: str | None,
        ingestion_id: str | None,
    ) -> list[PageContextResponse]:
        return self._store.page_contexts(
            source_pdf,
            pages,
            source_version_id,
            ingestion_id,
        )

    def browse_contexts(
        self,
        source_id: str,
        *,
        start_page: int | None,
        end_page: int | None,
        resume_page: int | None,
        scan_limit: int,
    ) -> SourceBrowseResponse | None:
        return self._store.browse_contexts(
            source_id,
            start_page=start_page,
            end_page=end_page,
            resume_page=resume_page,
            scan_limit=scan_limit,
        )

    def list_sources(
        self,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
        source_query: str | None = None,
    ) -> SourceListResponse:
        return self._store.list_sources(
            team_numbers=team_numbers,
            years=years,
            source_ids=source_ids,
            source_query=source_query,
        )


def create_mcp_server(
    settings: Settings,
    *,
    backend: RetrievalBackend | None = None,
    token_verifier: TokenVerifier | None = None,
) -> MCPServer:
    retrieval = backend or RagRetrievalBackend(settings)
    verifier = token_verifier or ClerkTokenVerifier(
        secret_key=settings.clerk_secret_key,
        issuer_url=str(settings.clerk_oauth_issuer_url),
    )
    server = MCPServer(
        name="mechbase",
        title="Mechbase FRC Mechanism Search",
        description="Search and retrieve mechanism details from FRC technical binders.",
        instructions=(
            "For every non-empty search or find_similar result, complete the visual selection "
            "flow before answering, even when the user does not explicitly ask for images. Call "
            "inspect_candidates with promising result ids and judge the actual page images plus "
            "page text against the request. Never present inspection images directly or use them "
            "as final answer images. If at least one image is relevant, call "
            "render_search_results with only those ids. If none are relevant, do not call the "
            "render tool and say that no useful image was found. Use fetch only when complete page "
            "text is needed. Use browse_source for multi-page evidence from the same binder; "
            "find_similar can return pages from other binders. All tools are read-only."
        ),
        version="0.3.0",
        auth=AuthSettings(
            issuer_url=settings.clerk_oauth_issuer_url,
            required_scopes=settings.mcp_required_scopes,
            resource_server_url=settings.mcp_endpoint_url,
        ),
        token_verifier=verifier,
    )
    public_base_url = str(settings.mcp_public_base_url)
    search_cache: TTLCache[str, SearchResponse] = TTLCache(ttl_seconds=15 * 60)
    similar_cache: TTLCache[str, SimilarPagesResponse | None] = TTLCache(ttl_seconds=60 * 60)
    source_cache: TTLCache[str, SourceListResponse] = TTLCache(ttl_seconds=5 * 60)
    parsed_public_url = urlparse(public_base_url)
    public_origin = f"{parsed_public_url.scheme}://{parsed_public_url.netloc}"

    @server.resource(
        SELECTED_RESULTS_WIDGET_URI,
        name="mechbase-selected-results",
        title="Selected Mechbase image results",
        description="Horizontal rail of FRC binder pages selected by the model.",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "prefersBorder": False,
                "domain": public_origin,
                "csp": {
                    "connectDomains": [],
                    "resourceDomains": [public_origin],
                },
            },
            "openai/widgetDescription": (
                "Shows only the FRC binder pages selected after visual inspection."
            ),
            "openai/widgetPrefersBorder": False,
            "openai/widgetDomain": public_origin,
            "openai/widgetCSP": {
                "connect_domains": [],
                "resource_domains": [public_origin],
                "redirect_domains": [public_origin],
            },
        },
    )
    def selected_results_widget() -> str:
        return SELECTED_RESULTS_WIDGET_HTML

    @server.tool(
        title="Search FRC mechanisms",
        description=(
            "Search FRC technical binders for mechanism designs. Use team_numbers, years, and "
            "source_ids whenever the user names exact teams, seasons, or binders. "
            "mechanism_types add semantic search terms; they are not exact metadata filters. "
            "The sort option orders relevant candidates and does not measure design quality or "
            "competition performance. Always follow a non-empty search with inspect_candidates "
            "on the most promising ids before answering. After visual review, display relevant "
            "pages only through render_search_results."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def search(
        query: Annotated[str, Field(max_length=500)],
        team_numbers: Annotated[list[TeamNumber], Field(max_length=20)] | None = None,
        years: Annotated[list[SeasonYear], Field(max_length=20)] | None = None,
        source_ids: Annotated[list[SourceId], Field(max_length=20)] | None = None,
        mechanism_types: Annotated[list[MechanismType], Field(max_length=8)] | None = None,
        sort: SearchSort = "relevance",
        top_k: Annotated[int | None, Field(ge=1, le=20)] = None,
    ) -> SearchOutput:
        filters = AppliedSearchFilters(
            team_numbers=[str(team) for team in dict.fromkeys(team_numbers or [])],
            years=list(dict.fromkeys(years or [])),
            source_ids=list(dict.fromkeys(source_ids or [])),
            mechanism_types=list(dict.fromkeys(mechanism_types or [])),
            sort=sort,
        )
        if not query.strip():
            return SearchOutput(
                results=[],
                applied_filters=filters,
                abstention_reason="The query was empty.",
            )
        request = SearchRequest(
            query=query.strip(),
            top_k=top_k or settings.mcp_search_top_k,
            team_numbers=filters.team_numbers,
            years=filters.years,
            source_ids=filters.source_ids,
            mechanism_types=filters.mechanism_types,
            sort=filters.sort,
        )
        cache_key = f"{retrieval.corpus_revision()}:{request.model_dump_json()}"
        response = search_cache.get_or_compute(cache_key, lambda: retrieval.search(request))
        return search_output(
            response,
            public_base_url,
            applied_filters=filters,
        )

    @server.tool(
        title="Inspect FRC mechanism candidate images",
        description=(
            "Inspect actual binder page images and extracted page text for result ids returned "
            "by search or find_similar. Compare every candidate against the user's request and "
            "discard irrelevant pages. Always follow visual review with render_search_results "
            "when at least one image is relevant. Never use inspection images as final answer "
            "images, cite them as displayed images, or describe them as shown to the user. They "
            "are model-only evaluation inputs. Render nothing when none are useful."
        ),
        annotations=READ_ONLY,
        meta={
            "openai/toolInvocation/invoking": "Reviewing candidate pages",
            "openai/toolInvocation/invoked": "Candidate pages reviewed",
        },
    )
    def inspect_candidates(
        ids: Annotated[list[ResultId], Field(min_length=1, max_length=6)],
    ) -> CallToolResult:
        candidates = []
        missing_ids = []
        content = [
            TextContent(
                type="text",
                text=(
                    "These candidate images are model-only evaluation inputs for visual review, not "
                    "answer images. Never show or cite them directly. Use each labeled page image "
                    "and its extracted text together. If any are relevant, you must call "
                    "render_search_results with only those ids before answering."
                ),
                annotations=MODEL_ONLY,
            )
        ]
        for result_id in dict.fromkeys(ids):
            context = retrieval.fetch(result_id)
            if context is None:
                missing_ids.append(result_id)
                continue

            preview = load_preview_image(context, settings)
            candidate = visual_candidate(
                context,
                result_id,
                public_base_url,
                has_image=preview is not None,
            )
            candidates.append(candidate)
            content.append(
                TextContent(
                    type="text",
                    text=(
                        f"Candidate id: {candidate.id}\n"
                        f"Title: {candidate.title}\n"
                        f"Image available: {'yes' if candidate.has_image else 'no'}\n"
                        f"Extracted page text:\n{candidate.text or '[No page text available]'}"
                    ),
                    annotations=MODEL_ONLY,
                )
            )
            if preview is not None:
                content.append(
                    ImageContent(
                        type="image",
                        data=base64.b64encode(preview.data).decode("ascii"),
                        mimeType=preview.mime_type,
                        annotations=MODEL_ONLY,
                    )
                )

        if not candidates:
            raise ValueError("None of the supplied result ids could be found.")

        output = InspectOutput(candidates=candidates, missing_ids=missing_ids)
        return CallToolResult(
            content=content,
            structuredContent=output.model_dump(mode="json"),
        )

    @server.tool(
        title="Fetch an FRC mechanism result",
        description=(
            "Retrieve the complete page text, source metadata, and citable image URLs for an id "
            "returned by search or find_similar. Use this only when full page detail is needed. "
            "Fetch does not replace inspect_candidates and must not be used to choose or display "
            "answer images."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def fetch(
        id: ResultId,
        adjacent_pages: Annotated[int, Field(ge=0, le=1)] = 0,
    ) -> FetchOutput:
        response = retrieval.fetch_contexts(id, adjacent_pages)
        if response is None:
            raise ValueError(f"No result found for id {id!r}.")
        return fetch_output(
            response.context,
            id,
            public_base_url,
            adjacent_contexts=response.adjacent_contexts,
        )

    @server.tool(
        title="Find similar mechanism pages",
        description=(
            "Find mechanism pages similar to a result returned by search. Use team_numbers, "
            "years, and source_ids when the user constrains the comparison. Similarity means "
            "related design evidence, not newer, better, or more successful. Always inspect "
            "promising returned ids before answering, then display only visually relevant pages."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def find_similar(
        id: ResultId,
        team_numbers: Annotated[list[TeamNumber], Field(max_length=20)] | None = None,
        years: Annotated[list[SeasonYear], Field(max_length=20)] | None = None,
        source_ids: Annotated[list[SourceId], Field(max_length=20)] | None = None,
        top_k: Annotated[int, Field(ge=1, le=20)] = 10,
    ) -> SimilarOutput:
        filters = AppliedSearchFilters(
            team_numbers=[str(team) for team in dict.fromkeys(team_numbers or [])],
            years=list(dict.fromkeys(years or [])),
            source_ids=list(dict.fromkeys(source_ids or [])),
        )
        cache_key = f"{retrieval.corpus_revision()}:{id}:{top_k}:{filters.model_dump_json()}"
        response = similar_cache.get_or_compute(
            cache_key,
            lambda: retrieval.find_similar(
                id,
                top_k,
                team_numbers=filters.team_numbers,
                years=filters.years,
                source_ids=filters.source_ids,
            ),
        )
        if response is None:
            raise ValueError(f"No result found for id {id!r}.")
        return similar_output(
            response,
            public_base_url,
            applied_filters=filters,
        )

    @server.tool(
        title="Display selected FRC mechanism pages",
        description=(
            "Display only visually relevant binder pages in an inline image rail. Always call "
            "inspect_candidates first, then pass only the ids whose images help answer the "
            "user's request. Do not use raw search ranking as the display decision."
        ),
        annotations=READ_ONLY,
        meta={
            "ui": {"resourceUri": SELECTED_RESULTS_WIDGET_URI},
            "openai/outputTemplate": SELECTED_RESULTS_WIDGET_URI,
            "openai/toolInvocation/invoking": "Preparing selected pages",
            "openai/toolInvocation/invoked": "Selected pages ready",
        },
        structured_output=True,
    )
    def render_search_results(
        ids: Annotated[list[ResultId], Field(min_length=1, max_length=8)],
    ) -> RenderOutput:
        contexts = []
        missing_ids = []
        for result_id in dict.fromkeys(ids):
            context = retrieval.fetch(result_id)
            if context is None:
                missing_ids.append(result_id)
            else:
                contexts.append((result_id, context))

        if missing_ids:
            missing = ", ".join(repr(result_id) for result_id in missing_ids)
            raise ValueError(f"No result found for id(s): {missing}.")

        output = render_output(contexts, public_base_url)
        if not output.results:
            raise ValueError("The selected results do not contain displayable images.")
        return output

    @server.tool(
        title="List indexed FRC binders",
        description=(
            "List indexed technical binders and their text and image coverage. Use exact team, "
            "year, or source-id filters to check whether Mechbase covers a request "
            "before substituting results from another team or season. source_query performs a "
            "case-insensitive filename and source-id substring match without an embedding call."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_sources(
        team_numbers: Annotated[list[TeamNumber], Field(max_length=50)] | None = None,
        years: Annotated[list[SeasonYear], Field(max_length=35)] | None = None,
        source_ids: Annotated[list[SourceId], Field(max_length=50)] | None = None,
        source_query: Annotated[str | None, Field(min_length=1, max_length=160)] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 50,
    ) -> SourceOutput:
        teams = [str(value) for value in dict.fromkeys(team_numbers or [])]
        normalized_years = list(dict.fromkeys(years or []))
        normalized_source_ids = list(dict.fromkeys(source_ids or []))
        normalized_source_query = source_query.strip() if source_query else None
        corpus_revision = retrieval.corpus_revision()
        cache_key = repr(
            (
                corpus_revision,
                teams,
                normalized_years,
                normalized_source_ids,
            )
        )
        response = source_cache.get_or_compute(
            cache_key,
            lambda: retrieval.list_sources(
                team_numbers=teams,
                years=normalized_years,
                source_ids=normalized_source_ids,
            ),
        )
        sources = _filter_source_query(response.sources, normalized_source_query)
        return source_output(
            sources[:limit],
            public_base_url,
            total_matching_sources=len(sources),
        )

    @server.tool(
        title="Browse an indexed FRC binder",
        description=(
            "Navigate one exact binder by page range or section without leaving that source. "
            "Use this for subsystem connections and designs that span several pages. Returned "
            "result ids can be passed to fetch, inspect_candidates, or render_search_results. "
            "When next_cursor is present, repeat the same section request with that complete "
            "cursor object to continue the bounded, generation-pinned scan."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def browse_source(
        source_id: SourceId,
        start_page: Annotated[int | None, Field(ge=1)] = None,
        end_page: Annotated[int | None, Field(ge=1)] = None,
        section: Annotated[
            str | None,
            Field(
                min_length=1,
                max_length=160,
                description="Case-insensitive section heading or source-text phrase.",
            ),
        ] = None,
        cursor: BrowseCursor | None = None,
        include_previews: bool = True,
    ) -> BrowseSourceOutput:
        if (start_page is None) != (end_page is None):
            raise ValueError("Provide both start_page and end_page, or neither.")
        if start_page is not None and end_page is not None:
            if end_page < start_page:
                raise ValueError("end_page must be greater than or equal to start_page.")
            if end_page - start_page + 1 > 10:
                raise ValueError("browse_source can return at most 10 pages per call.")
        if cursor is not None and start_page is not None:
            raise ValueError("cursor cannot be combined with an explicit page range.")
        section_needle = normalize_token_phrase(section) if section else None
        if section is not None and not section_needle:
            raise ValueError("section must contain at least one letter or number.")
        if cursor is not None and cursor.source_id != source_id:
            raise ValueError("The browse cursor belongs to a different source.")
        if cursor is not None and cursor.section != section_needle:
            raise ValueError(
                "The browse cursor belongs to a different section query; repeat the original "
                "section or restart browse_source without a cursor."
            )

        has_range = start_page is not None and end_page is not None
        response = retrieval.browse_contexts(
            source_id,
            start_page=start_page,
            end_page=end_page,
            resume_page=cursor.page if cursor is not None else None,
            scan_limit=BROWSE_SECTION_SCAN_PAGES if section_needle else 10,
        )
        if response is None:
            raise ValueError(f"No indexed source found for source_id {source_id!r}.")
        source = response.source
        if cursor is not None and (
            cursor.source_version_id != source.source_version_id
            or cursor.ingestion_id != source.ingestion_id
        ):
            raise ValueError(
                "The source generation changed after this browse cursor was issued; restart "
                "browse_source without a cursor."
            )

        available_pages = sorted(source.pages)
        requested_pages = response.requested_pages
        missing_pages = [page for page in requested_pages if page not in source.pages]
        pages_to_load = [page for page in requested_pages if page in source.pages]
        contexts = [context for context in response.contexts if context.page in pages_to_load]
        loaded_pages = {context.page for context in contexts}
        missing_pages.extend(page for page in pages_to_load if page not in loaded_pages)
        matching_contexts = [
            context
            for context in contexts
            if section_needle is None
            or contains_token_phrase(
                "\n".join((context.section or "", context.text)),
                section_needle,
            )
        ]
        total_matches = len(matching_contexts)
        matching_contexts = matching_contexts[:10]
        next_cursor = None
        if not has_range and pages_to_load:
            if total_matches > len(matching_contexts):
                resume_after = matching_contexts[-1].page
            else:
                resume_after = pages_to_load[-1]
            next_page = next(
                (page for page in available_pages if page > resume_after),
                None,
            )
            if next_page is not None:
                next_cursor = BrowseCursor(
                    page=next_page,
                    source_id=source_id,
                    section=section_needle,
                    source_version_id=source.source_version_id,
                    ingestion_id=source.ingestion_id,
                )

        return browse_source_output(
            source,
            matching_contexts,
            public_base_url,
            include_previews=include_previews,
            missing_pages=sorted(set(missing_pages)),
            scanned_pages=len(pages_to_load),
            next_cursor=next_cursor,
            truncated=next_cursor is not None,
        )

    return server


def _filter_source_query(
    sources: list[SourceSummary],
    source_query: str | None,
) -> list[SourceSummary]:
    if source_query is None:
        return sources
    needle = source_query.casefold()
    return [
        source
        for source in sources
        if needle in source.source_id.casefold()
        or needle in source.source_pdf.casefold()
        or needle in source.source_version_id.casefold()
    ]


def create_mcp_http_app(server: MCPServer, settings: Settings) -> ASGIApp:
    public_url = urlparse(settings.mcp_endpoint_url)
    public_host = public_url.netloc
    public_origin = f"{public_url.scheme}://{public_host}"
    allowed_hosts = [
        public_host,
        f"{public_url.hostname}:*",
        "127.0.0.1:*",
        "localhost:*",
        "testserver",
    ]
    allowed_origins = list(dict.fromkeys([public_origin, *settings.mcp_cors_origins]))
    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )
    return CORSMiddleware(
        mcp_app,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Authorization",
            "Content-Type",
            "Last-Event-ID",
            "Mcp-Protocol-Version",
            "Mcp-Session-Id",
        ],
        expose_headers=["Mcp-Session-Id", "WWW-Authenticate"],
    )
