from __future__ import annotations

import base64
from typing import Annotated, Protocol
from urllib.parse import urlparse

from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from app.mcp.auth import ClerkTokenVerifier
from app.mcp.images import load_preview_image
from app.mcp.results import (
    FetchOutput,
    InspectOutput,
    RenderOutput,
    SearchOutput,
    SourceOutput,
    fetch_output,
    render_output,
    search_output,
    source_output,
    visual_candidate,
)
from app.mcp.widget import SELECTED_RESULTS_WIDGET_HTML, SELECTED_RESULTS_WIDGET_URI
from app.rag.config import Settings
from app.rag.models import (
    ImageContextResponse,
    SearchResponse,
    SimilarPagesResponse,
    SourceListResponse,
)
from app.rag.search import search as rag_search
from app.rag.store import RagStore

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class RetrievalBackend(Protocol):
    def search(self, query: str, top_k: int) -> SearchResponse: ...

    def fetch(self, result_id: str) -> ImageContextResponse | None: ...

    def find_similar(self, result_id: str, top_k: int) -> SimilarPagesResponse | None: ...

    def list_sources(
        self,
        *,
        team: str | None,
        year: int | None,
        source: str | None,
    ) -> SourceListResponse: ...


class RagRetrievalBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def search(self, query: str, top_k: int) -> SearchResponse:
        return rag_search(query, top_k=top_k, debug=False, settings=self._settings)

    def fetch(self, result_id: str) -> ImageContextResponse | None:
        return RagStore(self._settings).image_context(result_id=result_id)

    def find_similar(self, result_id: str, top_k: int) -> SimilarPagesResponse | None:
        return RagStore(self._settings).similar_from_result_id(result_id, top_k)

    def list_sources(
        self,
        *,
        team: str | None,
        year: int | None,
        source: str | None,
    ) -> SourceListResponse:
        return RagStore(self._settings).list_sources(team=team, year=year, source=source)


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
            "Use search to find relevant FRC mechanism pages. Before displaying images, call "
            "inspect_candidates with promising result ids and judge the actual page images plus "
            "page text against the user's request. Call render_search_results only with visually "
            "relevant ids. If none are relevant, do not render them. Use fetch for complete page "
            "text and citations. All tools are read-only."
        ),
        version="0.1.0",
        auth=AuthSettings(
            issuer_url=settings.clerk_oauth_issuer_url,
            required_scopes=settings.mcp_required_scopes,
            resource_server_url=settings.mcp_endpoint_url,
        ),
        token_verifier=verifier,
    )
    public_base_url = str(settings.mcp_public_base_url)
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
            "Search FRC technical binders for mechanism designs and return stable result ids "
            "with citation URLs. Use fetch to retrieve the full page for a result."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def search(query: str) -> SearchOutput:
        if not query.strip():
            return SearchOutput(results=[])
        response = retrieval.search(query.strip(), settings.mcp_search_top_k)
        return search_output(response.results, public_base_url)

    @server.tool(
        title="Inspect FRC mechanism candidate images",
        description=(
            "Inspect actual binder page images and extracted page text for result ids returned "
            "by search or find_similar. Use this before render_search_results. Compare every "
            "candidate against the user's visual intent, discard irrelevant pages, and render "
            "nothing when none are useful. Inspect additional batches when needed."
        ),
        annotations=READ_ONLY,
        meta={
            "openai/toolInvocation/invoking": "Reviewing candidate pages",
            "openai/toolInvocation/invoked": "Candidate pages reviewed",
        },
    )
    def inspect_candidates(
        ids: Annotated[list[str], Field(min_length=1, max_length=6)],
    ) -> CallToolResult:
        candidates = []
        missing_ids = []
        content = [
            TextContent(
                type="text",
                text=(
                    "Visually review each labeled candidate. Use the page image and extracted "
                    "text together. Only pass relevant ids to render_search_results."
                ),
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
                )
            )
            if preview is not None:
                content.append(
                    ImageContent(
                        type="image",
                        data=base64.b64encode(preview.data).decode("ascii"),
                        mimeType=preview.mime_type,
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
            "returned by search or find_similar."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def fetch(id: str) -> FetchOutput:
        context = retrieval.fetch(id)
        if context is None:
            raise ValueError(f"No result found for id {id!r}.")
        return fetch_output(context, id, public_base_url)

    @server.tool(
        title="Find similar mechanism pages",
        description="Find mechanism pages similar to a result returned by search.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def find_similar(
        id: str,
        top_k: Annotated[int, Field(ge=1, le=20)] = 10,
    ) -> SearchOutput:
        response = retrieval.find_similar(id, top_k)
        if response is None:
            raise ValueError(f"No result found for id {id!r}.")
        return search_output(response.results, public_base_url)

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
        ids: Annotated[list[str], Field(min_length=1, max_length=8)],
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
        description="List indexed technical binders, optionally filtered by team, year, or filename.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_sources(
        team: str | None = None,
        year: int | None = None,
        source: str | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 50,
    ) -> SourceOutput:
        response = retrieval.list_sources(team=team, year=year, source=source)
        return source_output(response.sources[:limit], public_base_url)

    return server


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
