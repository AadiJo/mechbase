from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import anyio
import httpx
import httpx2
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from PIL import Image

from app.api.main import app
from app.mcp.auth import ClerkTokenVerifier
from app.mcp.images import load_preview_image
from app.mcp.results import fetch_output, search_output
from app.mcp.server import create_mcp_http_app, create_mcp_server
from app.rag.config import Settings, get_settings
from app.rag.models import (
    FetchContextResponse,
    ImageContextResponse,
    PageContextResponse,
    SearchCoverage,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarPagesResponse,
    SourceBrowseResponse,
    SourceListResponse,
    SourceSummary,
)


class FakeTokenVerifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token not in {"oauth_test", "oauth_limited"}:
            return None
        scopes = [] if token == "oauth_limited" else ["openid"]
        return AccessToken(
            token=token,
            client_id="chatgpt",
            scopes=scopes,
            subject="user_test",
            claims={"iss": "https://clerk.example.com"},
        )


class FakeRetrievalBackend:
    def __init__(self) -> None:
        self.search_calls: list[SearchRequest] = []
        self.fetch_calls: list[str] = []
        self.fetch_context_calls: list[dict[str, object]] = []
        self.similar_calls: list[dict[str, object]] = []
        self.similar_attempts: list[str] = []
        self.fetch_many_calls: list[list[str]] = []
        self.page_context_calls: list[dict[str, object]] = []
        self.list_source_calls: list[dict[str, object]] = []
        self.browse_context_calls: list[dict[str, object]] = []

    def search(self, request: SearchRequest) -> SearchResponse:
        self.search_calls.append(request)
        return SearchResponse(
            query=request.query,
            results=[_search_result("result_1", 12)],
            coverage=SearchCoverage(
                candidate_pages=1,
                candidate_sources=1,
                returned_pages=1,
            ),
        )

    def corpus_revision(self) -> str:
        return "test-corpus-v1"

    def fetch(self, result_id: str) -> ImageContextResponse | None:
        self.fetch_calls.append(result_id)
        if result_id == "missing":
            return None
        return ImageContextResponse(
            result_id=result_id,
            image_url="/images/254-2020/page-012/image-000.png",
            source_id="254-2020",
            source_version="version-a",
            source_version_id="254-2020@version-a",
            ingestion_id="ingestion-a",
            source_pdf="254-2020.pdf",
            team="254",
            year=2020,
            page=12,
            page_context_url="/pages/254-2020.pdf/12",
            page_text_url="/pages/254-2020.pdf/12/text",
            text="A compact two-stage elevator with a continuous belt rigging layout.",
            page_image_url="/images/254-2020/page-012/page.png",
            image_urls=["/images/254-2020/page-012/page.png"],
        )

    def fetch_contexts(
        self,
        result_id: str,
        adjacent_pages: int,
    ) -> FetchContextResponse | None:
        self.fetch_context_calls.append({"result_id": result_id, "adjacent_pages": adjacent_pages})
        context = self.fetch(result_id)
        if context is None:
            return None
        adjacent_contexts = []
        if adjacent_pages:
            adjacent_contexts = self.page_contexts(
                context.source_pdf,
                [context.page - 1, context.page + 1],
                context.source_version_id,
                context.ingestion_id,
            )
        return FetchContextResponse(
            context=context,
            adjacent_contexts=adjacent_contexts,
        )

    def fetch_many(self, result_ids: list[str]) -> dict[str, ImageContextResponse]:
        self.fetch_many_calls.append(result_ids)
        return {
            result_id: context
            for result_id in result_ids
            if (context := self.fetch(result_id)) is not None
        }

    def find_similar(
        self,
        result_id: str,
        top_k: int,
        *,
        team_numbers: list[str],
        years: list[int],
        source_ids: list[str],
    ) -> SimilarPagesResponse | None:
        self.similar_attempts.append(result_id)
        if result_id == "missing":
            return None
        self.similar_calls.append(
            {
                "result_id": result_id,
                "top_k": top_k,
                "team_numbers": team_numbers,
                "years": years,
                "source_ids": source_ids,
            }
        )
        result = _search_result("result_2", 8)
        result.debug["similarity_reason"] = "both"
        return SimilarPagesResponse(
            seed={"id": result_id, "modality": "text"},
            results=[result][:top_k],
            coverage=SearchCoverage(
                candidate_pages=1,
                candidate_sources=1,
                returned_pages=1,
            ),
        )

    def page_contexts(
        self,
        source_pdf: str,
        pages: list[int] | None,
        source_version_id: str | None,
        ingestion_id: str | None,
    ) -> list[PageContextResponse]:
        self.page_context_calls.append(
            {
                "source_pdf": source_pdf,
                "pages": pages,
                "source_version_id": source_version_id,
                "ingestion_id": ingestion_id,
            }
        )
        if source_pdf != "254-2020.pdf":
            return []
        available_pages = {1, 11, 12, 13}
        selected_pages = sorted({1, 12} if pages is None else available_pages & set(pages))
        return [
            PageContextResponse(
                source_id="254-2020",
                source_version="version-a",
                source_version_id=source_version_id,
                ingestion_id=ingestion_id,
                source_pdf=source_pdf,
                team="254",
                year=2020,
                page=page,
                section="Elevator" if page in {11, 12, 13} else "Overview",
                text=(
                    "Swerve Drive\nMechanical Design\nFloor-Pickup\nHarmonic drive\n机械臂设计"
                    if page == 1
                    else f"Page {page} source text"
                ),
                page_image_url=f"/images/254-2020/page-{page:03d}/page.png",
                image_urls=[f"/images/254-2020/page-{page:03d}/page.png"],
                result_ids=[f"result_page_{page}"],
                primary_result_id=f"result_page_{page}",
            )
            for page in selected_pages
        ]

    def list_sources(
        self,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
        source_query: str | None = None,
    ) -> SourceListResponse:
        self.list_source_calls.append(
            {
                "team_numbers": team_numbers,
                "years": years,
                "source_ids": source_ids,
                "source_query": source_query,
            }
        )
        sources = self._catalog_sources()
        needle = source_query.casefold() if source_query else None
        return SourceListResponse(
            sources=[
                source
                for source in sources
                if (not team_numbers or source.team in team_numbers)
                and (not years or source.year in years)
                and (not source_ids or source.source_id in source_ids)
                and (
                    needle is None
                    or needle in source.source_id.casefold()
                    or needle in source.source_pdf.casefold()
                )
            ]
        )

    @staticmethod
    def _catalog_sources() -> list[SourceSummary]:
        return [
            SourceSummary(
                source_id="254-2020",
                source_version="version-a",
                source_version_id="254-2020@version-a",
                ingestion_id="ingestion-a",
                source_pdf="254-2020.pdf",
                team="254",
                year=2020,
                pages=[1, 12],
                page_count=2,
                text_count=3,
                page_image_count=2,
                extracted_image_count=1,
                sample_image_urls=["/images/254-2020/page-012/page.png"],
            ),
            SourceSummary(
                source_id="4414-2024",
                source_version="version-b",
                source_version_id="4414-2024@version-b",
                ingestion_id="ingestion-b",
                source_pdf="4414-2024.pdf",
                team="4414",
                year=2024,
                pages=[4],
                page_count=1,
                text_count=1,
                page_image_count=1,
                extracted_image_count=0,
            ),
            SourceSummary(
                source_id="999-2024",
                source_version="version-c",
                source_version_id="999-2024@version-c",
                ingestion_id="ingestion-c",
                source_pdf="999-2024.pdf",
                team="999",
                year=2024,
                pages=list(range(1, 101)),
                page_count=100,
                text_count=100,
                page_image_count=100,
                extracted_image_count=0,
            ),
        ]

    def browse_contexts(
        self,
        source_id: str,
        *,
        start_page: int | None,
        end_page: int | None,
        resume_page: int | None,
        scan_limit: int,
    ) -> SourceBrowseResponse | None:
        self.browse_context_calls.append(
            {
                "source_id": source_id,
                "start_page": start_page,
                "end_page": end_page,
                "resume_page": resume_page,
                "scan_limit": scan_limit,
            }
        )
        source = next(
            (item for item in self._catalog_sources() if item.source_id == source_id),
            None,
        )
        if source is None:
            return None
        if start_page is not None and end_page is not None:
            requested_pages = list(range(start_page, end_page + 1))
        else:
            requested_pages = [
                page for page in sorted(source.pages) if resume_page is None or page >= resume_page
            ][:scan_limit]
        contexts = self.page_contexts(
            source.source_pdf,
            [page for page in requested_pages if page in source.pages],
            source.source_version_id,
            source.ingestion_id,
        )
        return SourceBrowseResponse(
            source=source,
            requested_pages=requested_pages,
            contexts=contexts,
        )


def _search_result(result_id: str, page: int) -> SearchResult:
    return SearchResult(
        id=result_id,
        score=0.9,
        score_band="strong",
        source_id="254-2020",
        source_version="version-a",
        source_version_id="254-2020@version-a",
        ingestion_id="ingestion-a",
        source_pdf="254-2020.pdf",
        team="254",
        year=2020,
        page=page,
        modality="text",
        text="elevator",
        artifact_path=None,
        artifact_url=None,
        linked_artifacts=[f"/tmp/page-{page}.png"],
        linked_artifact_urls=[f"/images/254-2020/page-{page:03d}/page.png"],
        page_context_url=f"/pages/254-2020.pdf/{page}",
        page_text_url=f"/pages/254-2020.pdf/{page}/text",
    )


def _settings(artifact_dir: Path | None = None) -> Settings:
    values = {
        "MCP_PUBLIC_BASE_URL": "https://api.example.com",
        "CLERK_OAUTH_ISSUER_URL": "https://clerk.example.com",
        "CLERK_SECRET_KEY": "clerk_secret",
    }
    if artifact_dir is not None:
        values["ARTIFACT_DIR"] = artifact_dir
    return Settings(
        **values,
    )


def test_search_output_preserves_weak_result_abstention() -> None:
    response = SearchResponse(
        query="cake recipe",
        results=[],
        coverage=SearchCoverage(candidate_pages=3, weak_pages_dropped=3),
        abstention_reason="No indexed pages met the calibrated relevance threshold.",
    )

    output = search_output(response, "https://api.example.com")

    assert output.results == []
    assert output.coverage.weak_pages_dropped == 3
    assert output.abstention_reason == ("No indexed pages met the calibrated relevance threshold.")


def test_fetch_citation_stays_on_the_exact_generation_artifact() -> None:
    def context(generation: str) -> ImageContextResponse:
        image_url = f"/images/254-2020/{generation}/page-001/page.png"
        return ImageContextResponse(
            result_id="result_1",
            source_id="254-2020",
            source_version_id=f"254-2020@{generation}",
            ingestion_id=generation,
            source_pdf="254-2020.pdf",
            page=1,
            page_context_url="/pages/254-2020.pdf/1",
            page_text_url="/pages/254-2020.pdf/1/text",
            text="elevator",
            page_image_url=image_url,
            image_urls=[image_url],
        )

    old_output = fetch_output(context("generation-old"), "result_1", "https://api.example.com")
    new_output = fetch_output(context("generation-new"), "result_1", "https://api.example.com")

    assert old_output.pages[0].url.endswith("/generation-old/page-001/page.png")
    assert new_output.pages[0].url.endswith("/generation-new/page-001/page.png")
    assert old_output.pages[0].url != new_output.pages[0].url


def test_candidate_preview_uses_shared_external_image_cache(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (64, 64), "white").save(source)
    context = ImageContextResponse(
        image_url="/images/page.png",
        source_pdf="254-2020.pdf",
        page=1,
        page_context_url="/pages/254-2020.pdf/1",
        page_text_url="/pages/254-2020.pdf/1/text",
        text="intake",
        image_urls=["/images/page.png"],
    )

    first = load_preview_image(context, _settings(tmp_path), max_side=32)
    second = load_preview_image(context, _settings(tmp_path), max_side=32)

    assert first is not None
    assert second == first
    assert len(list((tmp_path / ".embedding-cache").glob("*.jpg"))) == 1
    assert not (tmp_path / ".embed").exists()


def test_candidate_preview_rejects_a_decompression_bomb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def raise_decompression_bomb(*_args, **_kwargs):
        raise Image.DecompressionBombError

    source = tmp_path / "page.png"
    source.write_bytes(b"image")
    context = ImageContextResponse(
        image_url="/images/page.png",
        source_pdf="254-2020.pdf",
        page=1,
        page_context_url="/pages/254-2020.pdf/1",
        page_text_url="/pages/254-2020.pdf/1/text",
        text="intake",
        image_urls=["/images/page.png"],
    )
    monkeypatch.setattr(
        "app.mcp.images.cached_resized_image",
        raise_decompression_bomb,
    )

    assert load_preview_image(context, _settings(tmp_path)) is None


def test_clerk_token_verifier_accepts_active_tokens() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "clerk_idp_oauth_access_token",
                "id": "oat_test",
                "client_id": "client_test",
                "subject": "user_test",
                "scopes": ["openid"],
                "revoked": False,
                "revocation_reason": None,
                "expired": False,
                "expiration": 2_000_000_000,
                "created_at": 1_900_000_000,
                "updated_at": 1_900_000_000,
            },
        )

    async def verify() -> AccessToken | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            verifier = ClerkTokenVerifier(
                secret_key="sk_test",
                issuer_url="https://clerk.example.com",
                http_client=client,
            )
            return await verifier.verify_token("oauth_test")

    access_token = anyio.run(verify)

    assert access_token is not None
    assert access_token.client_id == "client_test"
    assert access_token.subject == "user_test"
    assert access_token.scopes == ["openid"]
    assert access_token.expires_at == 2_000_000_000
    assert requests[0].headers["authorization"] == "Bearer sk_test"
    assert json.loads(requests[0].content) == {"access_token": "oauth_test"}


def test_clerk_token_verifier_rejects_inactive_tokens() -> None:
    async def verify() -> AccessToken | None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"active": False})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            verifier = ClerkTokenVerifier(
                secret_key="sk_test",
                issuer_url="https://clerk.example.com",
                http_client=client,
            )
            return await verifier.verify_token("oauth_test")

    assert anyio.run(verify) is None


def test_mcp_discovery_is_public_and_api_keys_are_rejected() -> None:
    client = TestClient(app)
    settings = get_settings()

    metadata = client.get("/.well-known/oauth-protected-resource/mcp")
    assert metadata.status_code == 200
    assert metadata.json()["resource"] == settings.mcp_endpoint_url
    assert metadata.json()["authorization_servers"] == [
        str(settings.clerk_oauth_issuer_url).rstrip("/")
    ]

    response = client.post(
        "/mcp",
        headers={
            "Authorization": "Bearer mb_frontend_api_key",
            "Content-Type": "application/json",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
    )
    assert response.status_code == 401
    assert "resource_metadata=" in response.headers["www-authenticate"]


def test_mcp_protocol_lists_and_calls_read_only_tools(tmp_path: Path) -> None:
    page_image = tmp_path / "254-2020" / "page-012" / "page.png"
    page_image.parent.mkdir(parents=True)
    Image.new("RGB", (1800, 1200), "white").save(page_image)
    figure_image = page_image.parent / "image-000.png"
    Image.new("RGB", (640, 480), "blue").save(figure_image)

    async def exercise_server() -> None:
        settings = _settings(tmp_path)
        backend = FakeRetrievalBackend()
        server = create_mcp_server(
            settings,
            backend=backend,
            token_verifier=FakeTokenVerifier(),
        )
        http_app = create_mcp_http_app(server, settings)
        transport = httpx2.ASGITransport(app=http_app)
        async with server.session_manager.run():  # noqa: SIM117 - later contexts depend on prior values
            async with httpx2.AsyncClient(
                transport=transport,
                base_url="https://api.example.com",
                headers={"Authorization": "Bearer oauth_test"},
            ) as http_client:
                insufficient_scope = await http_client.post(
                    "/mcp",
                    headers={
                        "Authorization": "Bearer oauth_limited",
                        "Content-Type": "application/json",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"},
                        },
                    },
                )
                assert insufficient_scope.status_code == 403

                async with streamable_http_client(  # noqa: SIM117 - session uses streams
                    "https://api.example.com/mcp",
                    http_client=http_client,
                    terminate_on_close=False,
                ) as streams:
                    async with ClientSession(*streams) as session:
                        initialized = await session.initialize()
                        assert initialized.server_info.name == "mechbase"
                        assert "even when the user does not explicitly ask for images" in (
                            initialized.instructions or ""
                        )
                        assert "Never present inspection images directly" in (
                            initialized.instructions or ""
                        )

                        listed = await session.list_tools()
                        tools = {tool.name: tool for tool in listed.tools}
                        assert set(tools) == {
                            "search",
                            "inspect_candidates",
                            "fetch",
                            "find_similar",
                            "render_search_results",
                            "list_sources",
                            "browse_source",
                        }
                        assert set(tools["search"].input_schema["properties"]) == {
                            "query",
                            "team_numbers",
                            "years",
                            "source_ids",
                            "mechanism_types",
                            "sort",
                            "top_k",
                        }
                        assert "Always follow a non-empty search with inspect_candidates" in (
                            tools["search"].description or ""
                        )
                        assert tools["search"].output_schema is not None
                        assert set(tools["search"].output_schema["properties"]) == {
                            "results",
                            "applied_filters",
                            "coverage",
                            "abstention_reason",
                            "evidence_limits",
                        }
                        assert "source_query" in tools["list_sources"].input_schema["properties"]
                        assert set(tools["list_sources"].input_schema["properties"]) == {
                            "team_numbers",
                            "years",
                            "source_ids",
                            "source_query",
                            "limit",
                        }
                        assert tools["inspect_candidates"].output_schema is None
                        assert "Always follow visual review with render_search_results" in (
                            tools["inspect_candidates"].description or ""
                        )
                        assert "Never use inspection images as final answer images" in (
                            tools["inspect_candidates"].description or ""
                        )
                        assert set(tools["render_search_results"].input_schema["properties"]) == {
                            "ids",
                            "selections",
                        }
                        assert (
                            tools["render_search_results"].meta["ui"]["resourceUri"]
                            == "ui://mechbase/selected-results.html"
                        )
                        assert all(
                            tool.annotations.read_only_hint is True for tool in tools.values()
                        )
                        assert all(
                            tool.annotations.destructive_hint is False for tool in tools.values()
                        )

                        searched = await session.call_tool("search", {"query": " elevator "})
                        assert searched.is_error is False
                        assert searched.structured_content == {
                            "results": [
                                {
                                    "id": "result_1",
                                    "title": "Team 254: 254-2020.pdf, page 12",
                                    "url": (
                                        "https://api.example.com/images/254-2020/page-012/page.png"
                                    ),
                                    "source_id": "254-2020",
                                    "source_version": "version-a",
                                    "source_version_id": "254-2020@version-a",
                                    "ingestion_id": "ingestion-a",
                                    "source_pdf": "254-2020.pdf",
                                    "team": "254",
                                    "year": 2020,
                                    "page": 12,
                                    "snippet": "elevator",
                                    "score_band": "strong",
                                    "evidence": {
                                        "direct_source_text": True,
                                        "visible_image": True,
                                        "missing": [],
                                    },
                                }
                            ],
                            "applied_filters": {
                                "team_numbers": [],
                                "years": [],
                                "source_ids": [],
                                "mechanism_types": [],
                                "sort": "relevance",
                            },
                            "coverage": {
                                "candidate_pages": 1,
                                "candidate_sources": 1,
                                "weak_pages_dropped": 0,
                                "returned_pages": 1,
                                "candidate_window_truncated": False,
                            },
                            "abstention_reason": None,
                            "evidence_limits": [
                                (
                                    "Binder evidence does not independently verify competition "
                                    "performance."
                                ),
                                ("Vector relevance does not establish comparative design quality."),
                            ],
                        }
                        assert json.loads(searched.content[0].text) == searched.structured_content
                        assert len(backend.search_calls) == 1
                        assert backend.search_calls[0].query == "elevator"
                        assert backend.search_calls[0].top_k == 10
                        cached_search = await session.call_tool("search", {"query": " elevator "})
                        assert cached_search.is_error is False
                        assert len(backend.search_calls) == 1

                        filtered = await session.call_tool(
                            "search",
                            {
                                "query": "cone intake",
                                "team_numbers": [254, 4414, 254],
                                "years": [2023],
                                "source_ids": ["254-2023"],
                                "mechanism_types": ["intake"],
                                "sort": "newest",
                                "top_k": 4,
                            },
                        )
                        assert filtered.is_error is False
                        assert filtered.structured_content["applied_filters"] == {
                            "team_numbers": ["254", "4414"],
                            "years": [2023],
                            "source_ids": ["254-2023"],
                            "mechanism_types": ["intake"],
                            "sort": "newest",
                        }
                        assert backend.search_calls[1].team_numbers == ["254", "4414"]
                        assert backend.search_calls[1].years == [2023]
                        assert backend.search_calls[1].source_ids == ["254-2023"]
                        assert backend.search_calls[1].mechanism_types == ["intake"]
                        assert backend.search_calls[1].sort == "newest"
                        assert backend.search_calls[1].top_k == 4

                        inspected = await session.call_tool(
                            "inspect_candidates", {"ids": ["result_1"]}
                        )
                        assert inspected.is_error is False
                        assert inspected.structured_content["missing_ids"] == []
                        assert inspected.structured_content["candidates"][0]["id"] == "result_1"
                        assert inspected.structured_content["candidates"][0]["has_image"] is True
                        assert [
                            asset["asset_id"]
                            for asset in inspected.structured_content["candidates"][0]["assets"]
                        ] == ["page", "image-000"]
                        assert (
                            inspected.structured_content["candidates"][0]["assets"][1]["kind"]
                            == "figure"
                        )
                        assert (
                            inspected.structured_content["candidates"][0]["assets"][1]["width"]
                            == 640
                        )
                        assert backend.fetch_many_calls[-1] == ["result_1"]
                        image_blocks = [
                            block for block in inspected.content if block.type == "image"
                        ]
                        assert len(image_blocks) == 2
                        assert all(block.mime_type == "image/jpeg" for block in image_blocks)
                        assert all(
                            block.annotations is not None
                            and block.annotations.audience == ["assistant"]
                            for block in image_blocks
                        )
                        preview_sizes = []
                        for block in image_blocks:
                            with Image.open(BytesIO(base64.b64decode(block.data))) as preview_image:
                                assert preview_image.format == "JPEG"
                                preview_sizes.append(preview_image.size)
                        assert preview_sizes == [(1400, 933), (640, 480)]

                        inspected_page_only = await session.call_tool(
                            "inspect_candidates",
                            {"ids": ["result_1"], "include_assets": False},
                        )
                        assert inspected_page_only.is_error is False
                        assert (
                            len(
                                [
                                    block
                                    for block in inspected_page_only.content
                                    if block.type == "image"
                                ]
                            )
                            == 1
                        )

                        fetched = await session.call_tool("fetch", {"id": "result_1"})
                        assert fetched.is_error is False
                        assert fetched.structured_content["url"] == (
                            "https://api.example.com/images/254-2020/page-012/page.png"
                        )
                        assert "continuous belt" in fetched.structured_content["text"]
                        assert [page["page"] for page in fetched.structured_content["pages"]] == [
                            12
                        ]

                        fetched_with_neighbors = await session.call_tool(
                            "fetch", {"id": "result_1", "adjacent_pages": 1}
                        )
                        assert [
                            page["page"]
                            for page in fetched_with_neighbors.structured_content["pages"]
                        ] == [11, 12, 13]
                        assert all(
                            page["evidence"]["direct_source_text"]
                            for page in fetched_with_neighbors.structured_content["pages"]
                        )
                        assert all(
                            "/images/254-2020/" in page["url"]
                            for page in fetched_with_neighbors.structured_content["pages"]
                        )
                        assert backend.page_context_calls[-1]["pages"] == [11, 13]

                        fetch_context_calls_before_refresh = len(backend.fetch_context_calls)
                        refreshed_fetch = await session.call_tool(
                            "fetch", {"id": "result_1", "adjacent_pages": 1}
                        )
                        assert refreshed_fetch.is_error is False
                        assert [
                            page["page"] for page in refreshed_fetch.structured_content["pages"]
                        ] == [11, 12, 13]
                        assert (
                            len(backend.fetch_context_calls)
                            == fetch_context_calls_before_refresh + 1
                        )
                        assert backend.fetch_context_calls[-1] == {
                            "result_id": "result_1",
                            "adjacent_pages": 1,
                        }

                        similar = await session.call_tool(
                            "find_similar",
                            {
                                "id": "result_1",
                                "team_numbers": [254],
                                "years": [2020],
                                "source_ids": ["254-2020"],
                                "top_k": 1,
                            },
                        )
                        assert similar.is_error is False
                        assert similar.structured_content["results"][0]["id"] == "result_2"
                        assert similar.structured_content["coverage"]["candidate_pages"] == 1
                        assert similar.structured_content["coverage"]["returned_pages"] == 1
                        assert (
                            similar.structured_content["results"][0]["similarity_reason"] == "both"
                        )
                        assert backend.similar_calls[-1] == {
                            "result_id": "result_1",
                            "top_k": 1,
                            "team_numbers": ["254"],
                            "years": [2020],
                            "source_ids": ["254-2020"],
                        }
                        cached_similar = await session.call_tool(
                            "find_similar",
                            {
                                "id": "result_1",
                                "team_numbers": [254],
                                "years": [2020],
                                "source_ids": ["254-2020"],
                                "top_k": 1,
                            },
                        )
                        assert cached_similar.is_error is False
                        assert len(backend.similar_calls) == 1

                        missing_similar = await session.call_tool("find_similar", {"id": "missing"})
                        assert missing_similar.is_error is True
                        repeated_missing_similar = await session.call_tool(
                            "find_similar", {"id": "missing"}
                        )
                        assert repeated_missing_similar.is_error is True
                        assert backend.similar_attempts.count("missing") == 1

                        rendered = await session.call_tool(
                            "render_search_results", {"ids": ["result_1"]}
                        )
                        assert rendered.is_error is False
                        assert rendered.structured_content["results"] == [
                            {
                                "id": "result_1",
                                "asset_id": "page",
                                "title": "Team 254: 254-2020.pdf, page 12",
                                "url": (
                                    "https://api.example.com/images/254-2020/page-012/page.png"
                                ),
                                "image_url": (
                                    "https://api.example.com/images/254-2020/page-012/page.png"
                                ),
                                "source_pdf": "254-2020.pdf",
                                "team": "254",
                                "year": 2020,
                                "page": 12,
                            }
                        ]

                        rendered_figure = await session.call_tool(
                            "render_search_results",
                            {"selections": [{"id": "result_1", "asset_id": "image-000"}]},
                        )
                        assert rendered_figure.is_error is False
                        assert rendered_figure.structured_content["results"][0]["asset_id"] == (
                            "image-000"
                        )
                        assert rendered_figure.structured_content["results"][0][
                            "image_url"
                        ].endswith("/image-000.png")
                        assert (
                            "figure image-000"
                            in rendered_figure.structured_content["results"][0]["title"]
                        )

                        invalid_figure = await session.call_tool(
                            "render_search_results",
                            {"selections": [{"id": "result_1", "asset_id": "image-999"}]},
                        )
                        assert invalid_figure.is_error is True
                        ambiguous_render = await session.call_tool(
                            "render_search_results",
                            {
                                "ids": ["result_1"],
                                "selections": [{"id": "result_1"}],
                            },
                        )
                        assert ambiguous_render.is_error is True

                        resources = await session.list_resources()
                        widget = next(
                            resource
                            for resource in resources.resources
                            if str(resource.uri) == "ui://mechbase/selected-results.html"
                        )
                        assert widget.mime_type == "text/html;profile=mcp-app"
                        assert widget.meta["ui"]["csp"]["resourceDomains"] == [
                            "https://api.example.com"
                        ]
                        resource_contents = await session.read_resource(widget.uri)
                        assert "Selected mechanism pages" in resource_contents.contents[0].text
                        assert resource_contents.contents[0].meta["ui"]["prefersBorder"] is False

                        sources = await session.call_tool("list_sources", {"team_numbers": [254]})
                        assert sources.is_error is False
                        assert sources.structured_content["sources"][0]["team"] == "254"
                        assert sources.structured_content["sources"][0]["source_id"] == "254-2020"
                        assert (
                            sources.structured_content["sources"][0]["source_version_id"]
                            == "254-2020@version-a"
                        )
                        assert sources.structured_content["coverage_found"] is True
                        assert sources.structured_content["total_matching_sources"] == 1
                        assert sources.structured_content["truncated"] is False
                        assert sources.structured_content["sources"][0]["sample_image_urls"] == [
                            "https://api.example.com/images/254-2020/page-012/page.png"
                        ]
                        assert sources.structured_content["sources"][0]["text_count"] == 3
                        assert sources.structured_content["sources"][0]["page_image_count"] == 2
                        assert (
                            sources.structured_content["sources"][0]["extracted_image_count"] == 1
                        )
                        cached_sources = await session.call_tool(
                            "list_sources", {"team_numbers": [254]}
                        )
                        assert cached_sources.is_error is False
                        assert len(backend.list_source_calls) == 1

                        browsed = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "start_page": 1,
                                "end_page": 2,
                                "include_previews": True,
                            },
                        )
                        assert browsed.is_error is False
                        assert browsed.structured_content["source_id"] == "254-2020"
                        assert browsed.structured_content["pages"][0]["result_id"] == (
                            "result_page_1"
                        )
                        assert "result_ids" not in browsed.structured_content["pages"][0]
                        assert browsed.structured_content["missing_pages"] == [2]
                        assert browsed.structured_content["pages"][0]["image_url"].startswith(
                            "https://api.example.com/images/"
                        )
                        assert browsed.structured_content["pages"][0]["url"].startswith(
                            "https://api.example.com/images/"
                        )
                        assert browsed.structured_content["scanned_pages"] == 1
                        assert browsed.structured_content["next_cursor"] is None
                        assert backend.page_context_calls[-1]["pages"] == [1]
                        assert backend.browse_context_calls[-1] == {
                            "source_id": "254-2020",
                            "start_page": 1,
                            "end_page": 2,
                            "resume_page": None,
                            "scan_limit": 10,
                        }

                        browsed_section = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "section": "elevator",
                                "include_previews": False,
                            },
                        )
                        assert browsed_section.is_error is False
                        assert [
                            page["page"] for page in browsed_section.structured_content["pages"]
                        ] == [12]
                        assert browsed_section.structured_content["pages"][0]["image_url"] is None
                        assert backend.page_context_calls[-1]["pages"] == [1, 12]

                        browsed_text_heading = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "section": "swerve",
                                "include_previews": False,
                            },
                        )
                        assert browsed_text_heading.is_error is False
                        assert [
                            page["page"]
                            for page in browsed_text_heading.structured_content["pages"]
                        ] == [1]

                        browsed_hyphenated_heading = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "section": "floor pickup",
                                "include_previews": False,
                            },
                        )
                        assert browsed_hyphenated_heading.is_error is False
                        assert [
                            page["page"]
                            for page in browsed_hyphenated_heading.structured_content["pages"]
                        ] == [1]

                        browsed_substring = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "section": "arm",
                                "include_previews": False,
                            },
                        )
                        assert browsed_substring.is_error is False
                        assert browsed_substring.structured_content["pages"] == []

                        browsed_non_latin_heading = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "section": "机械臂",
                                "include_previews": False,
                            },
                        )
                        assert browsed_non_latin_heading.is_error is False
                        assert [
                            page["page"]
                            for page in browsed_non_latin_heading.structured_content["pages"]
                        ] == [1]

                        bounded_section_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "999-2024",
                                "section": "elevator",
                            },
                        )
                        assert bounded_section_browse.is_error is False
                        assert bounded_section_browse.structured_content["scanned_pages"] == 50
                        assert bounded_section_browse.structured_content["next_cursor"] == {
                            "page": 51,
                            "source_id": "999-2024",
                            "section": "elevator",
                            "source_version_id": "999-2024@version-c",
                            "ingestion_id": "ingestion-c",
                        }
                        assert bounded_section_browse.structured_content["truncated"] is True
                        assert backend.page_context_calls[-1]["pages"] == list(range(1, 51))

                        expanding_section_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "999-2024",
                                "section": "ß" * 100,
                            },
                        )
                        assert expanding_section_browse.is_error is False
                        assert (
                            expanding_section_browse.structured_content["next_cursor"]["section"]
                            == "ss" * 100
                        )
                        omitted_section_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "999-2024",
                                "cursor": bounded_section_browse.structured_content["next_cursor"],
                            },
                        )
                        assert omitted_section_browse.is_error is True
                        changed_section_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "999-2024",
                                "section": "intake",
                                "cursor": bounded_section_browse.structured_content["next_cursor"],
                            },
                        )
                        assert changed_section_browse.is_error is True
                        stale_cursor_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "999-2024",
                                "section": "elevator",
                                "cursor": {
                                    "page": 51,
                                    "source_id": "999-2024",
                                    "section": "elevator",
                                    "source_version_id": "999-2024@old",
                                    "ingestion_id": "ingestion-old",
                                },
                            },
                        )
                        assert stale_cursor_browse.is_error is True
                        continued_section_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "999-2024",
                                "section": "elevator",
                                "cursor": bounded_section_browse.structured_content["next_cursor"],
                            },
                        )
                        assert continued_section_browse.is_error is False
                        assert continued_section_browse.structured_content["next_cursor"] is None
                        assert backend.page_context_calls[-1]["pages"] == list(range(51, 101))

                        oversized_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "start_page": 1,
                                "end_page": 11,
                            },
                        )
                        assert oversized_browse.is_error is True
                        invalid_adjacent_fetch = await session.call_tool(
                            "fetch", {"id": "result_1", "adjacent_pages": 2}
                        )
                        assert invalid_adjacent_fetch.is_error is True
                        incomplete_browse = await session.call_tool(
                            "browse_source",
                            {"source_id": "254-2020", "start_page": 1},
                        )
                        assert incomplete_browse.is_error is True
                        queried_sources = await session.call_tool(
                            "list_sources",
                            {"team_numbers": [254], "source_query": "2020"},
                        )
                        assert queried_sources.is_error is False
                        assert queried_sources.structured_content["coverage_found"] is True
                        assert len(backend.list_source_calls) == 1

                        filtered_sources = await session.call_tool(
                            "list_sources",
                            {
                                "team_numbers": [4414],
                                "years": [2024],
                                "source_ids": ["4414-2024"],
                                "source_query": "4414",
                            },
                        )
                        assert filtered_sources.is_error is False
                        assert filtered_sources.structured_content["sources"][0]["team"] == "4414"
                        assert filtered_sources.structured_content["sources"][0]["year"] == 2024
                        assert len(backend.list_source_calls) == 2
                        assert backend.list_source_calls[-1] == {
                            "team_numbers": ["4414"],
                            "years": [2024],
                            "source_ids": ["4414-2024"],
                            "source_query": None,
                        }

                        browse_context_calls_before_refresh = len(backend.browse_context_calls)
                        refreshed_browse = await session.call_tool(
                            "browse_source",
                            {
                                "source_id": "254-2020",
                                "start_page": 1,
                                "end_page": 2,
                            },
                        )
                        assert refreshed_browse.is_error is False
                        assert refreshed_browse.structured_content["missing_pages"] == [2]
                        assert (
                            len(backend.browse_context_calls)
                            == browse_context_calls_before_refresh + 1
                        )
                        assert backend.browse_context_calls[-1] == {
                            "source_id": "254-2020",
                            "start_page": 1,
                            "end_page": 2,
                            "resume_page": None,
                            "scan_limit": 10,
                        }

                        invalid = await session.call_tool(
                            "find_similar", {"id": "result_1", "top_k": 21}
                        )
                        assert invalid.is_error is True

                        oversized_id = "x" * 257
                        invalid_id = await session.call_tool("find_similar", {"id": oversized_id})
                        assert invalid_id.is_error is True
                        assert oversized_id not in backend.similar_attempts

                        too_many_images = await session.call_tool(
                            "inspect_candidates",
                            {"ids": [f"result_{index}" for index in range(7)]},
                        )
                        assert too_many_images.is_error is True

                        missing = await session.call_tool("fetch", {"id": "missing"})
                        assert missing.is_error is True

                        missing_render = await session.call_tool(
                            "render_search_results", {"ids": ["missing"]}
                        )
                        assert missing_render.is_error is True

    anyio.run(exercise_server)
