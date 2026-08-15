from __future__ import annotations

import json

import anyio
import httpx
import httpx2
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken

from app.api.main import app
from app.mcp.auth import ClerkTokenVerifier
from app.mcp.server import create_mcp_http_app, create_mcp_server
from app.rag.config import Settings
from app.rag.models import (
    ImageContextResponse,
    SearchResponse,
    SearchResult,
    SimilarPagesResponse,
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
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> SearchResponse:
        self.search_calls.append((query, top_k))
        return SearchResponse(query=query, results=[_search_result("result_1", 12)])

    def fetch(self, result_id: str) -> ImageContextResponse | None:
        if result_id == "missing":
            return None
        return ImageContextResponse(
            result_id=result_id,
            image_url="/images/254-2020/page-012/image-000.png",
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

    def find_similar(self, result_id: str, top_k: int) -> SimilarPagesResponse | None:
        if result_id == "missing":
            return None
        return SimilarPagesResponse(
            seed={"id": result_id},
            results=[_search_result("result_2", 8)][:top_k],
        )

    def list_sources(
        self,
        *,
        team: str | None,
        year: int | None,
        source: str | None,
    ) -> SourceListResponse:
        return SourceListResponse(
            sources=[
                SourceSummary(
                    source_pdf=source or "254-2020.pdf",
                    team=team or "254",
                    year=year or 2020,
                    pages=[1, 12],
                    page_count=2,
                    sample_image_urls=["/images/254-2020/page-012/page.png"],
                )
            ]
        )


def _search_result(result_id: str, page: int) -> SearchResult:
    return SearchResult(
        id=result_id,
        score=0.9,
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


def _settings() -> Settings:
    return Settings(
        MCP_PUBLIC_BASE_URL="https://api.example.com",
        CLERK_OAUTH_ISSUER_URL="https://clerk.example.com",
        CLERK_SECRET_KEY="clerk_secret",
    )


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

    metadata = client.get("/.well-known/oauth-protected-resource/mcp")
    assert metadata.status_code == 200
    assert metadata.json()["resource"] == "https://api-frcrag-v2.johari-dev.com/mcp"
    assert metadata.json()["authorization_servers"] == ["https://clerk.mechbase.johari-dev.com"]

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


def test_mcp_protocol_lists_and_calls_read_only_tools() -> None:
    async def exercise_server() -> None:
        settings = _settings()
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

                        listed = await session.list_tools()
                        tools = {tool.name: tool for tool in listed.tools}
                        assert set(tools) == {"search", "fetch", "find_similar", "list_sources"}
                        assert set(tools["search"].input_schema["properties"]) == {"query"}
                        assert tools["search"].output_schema is not None
                        assert set(tools["search"].output_schema["properties"]) == {"results"}
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
                                }
                            ]
                        }
                        assert json.loads(searched.content[0].text) == searched.structured_content
                        assert backend.search_calls == [("elevator", 10)]

                        fetched = await session.call_tool("fetch", {"id": "result_1"})
                        assert fetched.is_error is False
                        assert fetched.structured_content["url"] == (
                            "https://api.example.com/images/254-2020/page-012/page.png"
                        )
                        assert "continuous belt" in fetched.structured_content["text"]

                        similar = await session.call_tool(
                            "find_similar", {"id": "result_1", "top_k": 1}
                        )
                        assert similar.is_error is False
                        assert similar.structured_content["results"][0]["id"] == "result_2"

                        sources = await session.call_tool("list_sources", {"team": "254"})
                        assert sources.is_error is False
                        assert sources.structured_content["sources"][0]["team"] == "254"
                        assert sources.structured_content["sources"][0]["sample_image_urls"] == [
                            "https://api.example.com/images/254-2020/page-012/page.png"
                        ]

                        invalid = await session.call_tool(
                            "find_similar", {"id": "result_1", "top_k": 21}
                        )
                        assert invalid.is_error is True

                        missing = await session.call_tool("fetch", {"id": "missing"})
                        assert missing.is_error is True

    anyio.run(exercise_server)
