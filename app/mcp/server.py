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
from app.mcp.contexts import (
    GameContextOutput,
    GameTopic,
    TeamContextOutput,
    game_context,
    team_research_targets,
    team_web_queries,
)
from app.mcp.images import (
    CandidateAssetSource,
    PreviewImage,
    candidate_asset_sources,
    load_preview_url,
)
from app.mcp.results import (
    MAX_NORMALIZED_SECTION_LENGTH,
    AppliedSearchFilters,
    BrowseCursor,
    BrowseSourceOutput,
    FetchOutput,
    InspectOutput,
    RenderOutput,
    RenderSelection,
    SearchOutput,
    SimilarOutput,
    SourceOutput,
    browse_source_output,
    fetch_output,
    render_output,
    search_output,
    similar_output,
    source_output,
    visual_asset,
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
MAX_INSPECTED_FIGURES = 4
MAX_INSPECTION_IMAGES = 12
MAX_INSPECTION_ENCODED_BYTES = 8 * 1024 * 1024


class RetrievalBackend(Protocol):
    def search(self, request: SearchRequest) -> SearchResponse: ...

    def fetch(self, result_id: str) -> ImageContextResponse | None: ...

    def fetch_contexts(
        self,
        result_id: str,
        adjacent_pages: int,
    ) -> FetchContextResponse | None: ...

    def fetch_many(self, result_ids: list[str]) -> dict[str, ImageContextResponse]: ...

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

    def fetch_many(self, result_ids: list[str]) -> dict[str, ImageContextResponse]:
        return self._store.image_contexts(result_ids)

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
            "render_search_results with the inspected result and asset ids. If none "
            "are relevant, do not call the render tool and say that no useful image was found. "
            "Use fetch only when complete page text is needed. Use browse_source for multi-page "
            "evidence from the same binder; find_similar can return pages from other binders. Use "
            "get_game_context to ground season terminology. Use get_team_context for exact team "
            "coverage and browse its public targets before making performance claims. All tools "
            "are read-only."
        ),
        version="0.5.0",
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
            "Inspect actual binder page images, up to four extracted figures per page, and page "
            "text for result ids returned by search or find_similar. Stable asset ids identify "
            "both page and figure assets for render_search_results. Compare every candidate "
            "against the user's request and discard irrelevant images. Always follow visual review with "
            "render_search_results when at least one image is relevant. Never use inspection "
            "images as final answer images, cite them, or describe them as shown to the user. They "
            "are model-only inputs. When truncated_ids is non-empty, inspect those ids again in "
            "smaller batches before deciding relevance. Render nothing when none are useful."
        ),
        annotations=READ_ONLY,
        meta={
            "openai/toolInvocation/invoking": "Reviewing candidate pages",
            "openai/toolInvocation/invoked": "Candidate pages reviewed",
        },
    )
    def inspect_candidates(
        ids: Annotated[list[ResultId], Field(min_length=1, max_length=6)],
        include_assets: bool = True,
    ) -> CallToolResult:
        unique_ids = list(dict.fromkeys(ids))
        contexts = retrieval.fetch_many(unique_ids)
        candidates = []
        missing_ids = []
        content = [
            TextContent(
                type="text",
                text=(
                    "These candidate images are model-only evaluation inputs for visual review, not "
                    "answer images. Never show or cite them directly. Use each labeled page image "
                    "and its extracted text together. If any are relevant, you must call "
                    "render_search_results with only those result and asset ids before "
                    "answering."
                ),
                annotations=MODEL_ONLY,
            )
        ]
        contexts_by_id: dict[str, ImageContextResponse] = {}
        valid_assets_by_id: dict[
            str,
            list[tuple[CandidateAssetSource, PreviewImage]],
        ] = {}
        for result_id in unique_ids:
            context = contexts.get(result_id)
            if context is None:
                missing_ids.append(result_id)
                continue

            contexts_by_id[result_id] = context
            valid_assets = []
            loaded_figures = 0
            for asset_source in candidate_asset_sources(
                context,
                include_assets=include_assets,
            ):
                if asset_source.kind == "figure" and loaded_figures == (
                    MAX_INSPECTED_FIGURES if include_assets else 1
                ):
                    break
                preview = load_preview_url(asset_source.image_url, settings)
                if preview is None:
                    continue
                if asset_source.kind == "figure":
                    loaded_figures += 1
                valid_assets.append((asset_source, preview))
            valid_assets_by_id[result_id] = valid_assets

        if len(valid_assets_by_id) == 1:
            result_id, valid_assets = next(iter(valid_assets_by_id.items()))
            encoded_bytes = sum(
                4 * ((len(preview.data) + 2) // 3) for _source, preview in valid_assets
            )
            for max_side in (1200, 1000, 800, 640, 480, 320):
                if encoded_bytes <= MAX_INSPECTION_ENCODED_BYTES:
                    break
                resized_assets = [
                    (asset_source, preview)
                    for asset_source, _preview in valid_assets
                    if (
                        preview := load_preview_url(
                            asset_source.image_url,
                            settings,
                            max_side=max_side,
                        )
                    )
                    is not None
                ]
                valid_assets = resized_assets
                encoded_bytes = sum(
                    4 * ((len(preview.data) + 2) // 3) for _asset_source, preview in valid_assets
                )
            valid_assets_by_id[result_id] = valid_assets

        selected_by_id: dict[str, list[tuple[CandidateAssetSource, PreviewImage]]] = {
            result_id: [] for result_id in contexts_by_id
        }
        selected_asset_ids: set[str] = set()
        selected_image_count = 0
        selected_encoded_bytes = 0

        def select_asset(
            result_id: str,
            asset: tuple[CandidateAssetSource, PreviewImage],
        ) -> bool:
            nonlocal selected_encoded_bytes, selected_image_count
            asset_source, preview = asset
            if asset_source.asset_id in selected_asset_ids:
                return False
            encoded_bytes = 4 * ((len(preview.data) + 2) // 3)
            if (
                selected_image_count == MAX_INSPECTION_IMAGES
                or selected_encoded_bytes + encoded_bytes > MAX_INSPECTION_ENCODED_BYTES
            ):
                return False
            selected_by_id[result_id].append(asset)
            selected_asset_ids.add(asset_source.asset_id)
            selected_image_count += 1
            selected_encoded_bytes += encoded_bytes
            return True

        # Give every candidate's full page first priority, then fall back to its first figure.
        for result_id, valid_assets in valid_assets_by_id.items():
            page_asset = next(
                (asset for asset in valid_assets if asset[0].kind == "page"),
                None,
            )
            if page_asset is not None:
                select_asset(result_id, page_asset)
        for result_id, valid_assets in valid_assets_by_id.items():
            if not selected_by_id[result_id] and valid_assets:
                fallback_figure = next(
                    (asset for asset in valid_assets if asset[0].kind == "figure"),
                    None,
                )
                if fallback_figure is not None:
                    select_asset(result_id, fallback_figure)

        # Add figures round-robin so one candidate cannot consume the call-wide budget.
        for figure_index in range(MAX_INSPECTED_FIGURES):
            for result_id, valid_assets in valid_assets_by_id.items():
                figures = [asset for asset in valid_assets if asset[0].kind == "figure"]
                if figure_index < len(figures):
                    select_asset(result_id, figures[figure_index])

        truncated_ids = [
            result_id
            for result_id, valid_assets in valid_assets_by_id.items()
            if len(selected_by_id[result_id]) < len(valid_assets)
        ]
        if truncated_ids:
            content.append(
                TextContent(
                    type="text",
                    text=(
                        "The call-wide image budget omitted one or more previews for candidate "
                        f"ids: {', '.join(truncated_ids)}. Re-run inspect_candidates with these "
                        "ids in smaller batches before deciding whether their images are relevant."
                    ),
                    annotations=MODEL_ONLY,
                )
            )

        for result_id, context in contexts_by_id.items():
            selected_assets = selected_by_id[result_id]
            loaded_assets = [
                visual_asset(context, asset_source, preview, public_base_url)
                for asset_source, preview in selected_assets
            ]
            candidate = visual_candidate(
                context,
                result_id,
                public_base_url,
                assets=loaded_assets,
            )
            candidates.append(candidate)
            if candidate.has_image:
                image_availability = "yes"
            elif result_id in truncated_ids:
                image_availability = "deferred by call budget"
            else:
                image_availability = "no"
            content.append(
                TextContent(
                    type="text",
                    text=(
                        f"Candidate id: {candidate.id}\n"
                        f"Title: {candidate.title}\n"
                        f"Image available: {image_availability}\n"
                        f"Extracted page text:\n{candidate.text or '[No page text available]'}"
                    ),
                    annotations=MODEL_ONLY,
                )
            )
            for asset_source, preview in selected_assets:
                content.append(
                    TextContent(
                        type="text",
                        text=(
                            f"Candidate id: {candidate.id}\n"
                            f"Asset id: {asset_source.asset_id}\n"
                            f"Asset kind: {asset_source.kind}\n"
                            f"Original dimensions: {preview.width}x{preview.height}\n"
                            "Use this asset only if it visually helps answer the request."
                        ),
                        annotations=MODEL_ONLY,
                    )
                )
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

        output = InspectOutput(
            candidates=candidates,
            missing_ids=missing_ids,
            truncated_ids=truncated_ids,
        )
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
            "Display only visually relevant binder pages or extracted figures in an inline image "
            "rail. Always call inspect_candidates first. Current clients must use selections and "
            "pass the inspected asset_id for every page or figure. This rejects selections made "
            "against an older source generation. The legacy ids input remains a best-effort "
            "full-page alias and cannot detect a source refresh. Provide ids or selections, never "
            "both."
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
        ids: Annotated[list[ResultId] | None, Field(min_length=1, max_length=8)] = None,
        selections: Annotated[
            list[RenderSelection] | None,
            Field(min_length=1, max_length=8),
        ] = None,
    ) -> RenderOutput:
        if (ids is None) == (selections is None):
            raise ValueError("Provide either ids or selections, but not both.")
        requested = (
            [(result_id, None) for result_id in ids]
            if ids is not None
            else [(selection.id, selection.asset_id) for selection in selections or []]
        )
        requested = list(
            {
                (result_id, asset_id): (result_id, asset_id) for result_id, asset_id in requested
            }.values()
        )
        contexts_by_id = retrieval.fetch_many(
            list(dict.fromkeys(result_id for result_id, _asset_id in requested))
        )
        missing_ids = []
        resolved = []
        missing_assets = []
        for result_id, asset_id in requested:
            context = contexts_by_id.get(result_id)
            if context is None:
                missing_ids.append(result_id)
                continue
            available_assets = candidate_asset_sources(context, include_assets=True)
            selected_asset = (
                next(
                    (asset for asset in available_assets if asset.asset_id == asset_id),
                    None,
                )
                if asset_id is not None
                else next(iter(available_assets), None)
            )
            if selected_asset is None:
                missing_assets.append((result_id, asset_id))
                continue
            if load_preview_url(selected_asset.image_url, settings) is None:
                missing_assets.append((result_id, asset_id))
                continue
            resolved.append(
                (
                    result_id,
                    selected_asset.asset_id,
                    selected_asset.kind,
                    context,
                    selected_asset.image_url,
                )
            )

        if missing_ids:
            missing = ", ".join(repr(result_id) for result_id in dict.fromkeys(missing_ids))
            raise ValueError(f"No result found for id(s): {missing}.")
        if missing_assets:
            missing = ", ".join(
                f"{asset_id!r} for {result_id!r}" for result_id, asset_id in missing_assets
            )
            raise ValueError(f"No inspected asset found for selection(s): {missing}.")

        output = render_output(resolved, public_base_url)
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
        if section_needle is not None and len(section_needle) > MAX_NORMALIZED_SECTION_LENGTH:
            raise ValueError(
                "section expands beyond the normalized query limit; use a shorter phrase."
            )
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

    @server.tool(
        title="Get reviewed FRC game context",
        description=(
            "Get reviewed, structured season terminology and mechanism-relevant game context "
            "with citations to the official FIRST manual. Use this before interpreting historical "
            "binder language or comparing mechanisms across games. Request only the topics needed. "
            "This summary is not a substitute for the official manual and explicitly reports "
            "unsupported seasons or missing topics."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_game_context(
        year: SeasonYear,
        topics: Annotated[list[GameTopic] | None, Field(max_length=6)] = None,
    ) -> GameContextOutput:
        return game_context(year, list(dict.fromkeys(topics or [])))

    @server.tool(
        title="Get exact FRC team context",
        description=(
            "Resolve exact indexed binder coverage for one FRC team and optional season. When a "
            "mechanism query is supplied, search only that team's matching indexed sources. This "
            "tool does not fetch live competition performance. If the user asks about records, "
            "rankings, awards, matches, or performance, browse and cite the returned public FIRST "
            "Events and The Blue Alliance targets before answering. Do not infer performance from "
            "binder content or infer mechanism causality from event results."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_team_context(
        team_number: TeamNumber,
        year: SeasonYear | None = None,
        mechanism_query: Annotated[str | None, Field(min_length=1, max_length=500)] = None,
        top_k: Annotated[int, Field(ge=1, le=20)] = 8,
    ) -> TeamContextOutput:
        team = str(team_number)
        years = [year] if year is not None else []
        corpus_revision = retrieval.corpus_revision()
        cache_key = repr((corpus_revision, [team], years, []))
        catalog = source_cache.get_or_compute(
            cache_key,
            lambda: retrieval.list_sources(
                team_numbers=[team],
                years=years,
                source_ids=[],
            ),
        )
        matching_sources = catalog.sources
        indexed_sources = source_output(
            matching_sources,
            public_base_url,
            total_matching_sources=len(matching_sources),
        )
        mechanism_search = None
        if mechanism_query is not None:
            query = mechanism_query.strip()
            if not query:
                raise ValueError("mechanism_query must contain non-whitespace text.")
            filters = AppliedSearchFilters(
                team_numbers=[team],
                years=[year] if year is not None else [],
            )
            if matching_sources:
                request = SearchRequest(
                    query=query,
                    top_k=top_k,
                    team_numbers=filters.team_numbers,
                    years=filters.years,
                )
                cache_key = f"{corpus_revision}:{request.model_dump_json()}"
                response = search_cache.get_or_compute(
                    cache_key,
                    lambda: retrieval.search(request),
                )
                mechanism_search = search_output(
                    response,
                    public_base_url,
                    applied_filters=filters,
                )
            else:
                mechanism_search = SearchOutput(
                    results=[],
                    applied_filters=filters,
                    abstention_reason=(
                        "No indexed binder matches the requested team and season, so mechanism "
                        "search was not run."
                    ),
                )

        return TeamContextOutput(
            team_number=team_number,
            year=year,
            indexed_sources=indexed_sources,
            mechanism_search=mechanism_search,
            live_research_targets=team_research_targets(team_number, year),
            suggested_web_queries=team_web_queries(team_number, year),
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
