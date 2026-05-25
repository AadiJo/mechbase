from fastapi.testclient import TestClient

import app.api.auth as auth
import app.api.main as main
from app.api.auth import ApiKeyContext
from app.api.main import app
from app.rag.config import Settings
from app.rag.models import SearchResponse


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_search_requires_api_key() -> None:
    client = TestClient(app)
    response = client.post("/search", json={"query": "shooter"})
    assert response.status_code == 401


def test_validate_api_key_accepts_valid_key(monkeypatch) -> None:
    def fake_validate_mechbase_api_key(value, settings=None):
        return ApiKeyContext(
            api_key_id="api_key_test",
            organization_id="workspace_test",
            permissions=("search:read",),
        )

    monkeypatch.setattr(auth, "validate_mechbase_api_key", fake_validate_mechbase_api_key)
    client = TestClient(app)
    response = client.get("/auth/validate", headers={"Authorization": "Bearer sk_test"})

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "apiKeyId": "api_key_test",
        "workspaceId": "workspace_test",
        "permissions": ["search:read"],
    }


def test_rate_limit_defaults_to_20_requests() -> None:
    settings = Settings()
    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_max_requests == 20
    assert settings.rate_limit_window_seconds == 60


def test_validate_api_key_sends_rate_limit_config(monkeypatch) -> None:
    posted = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {
                "valid": True,
                "apiKeyId": "api_key_test",
                "workspaceId": "workspace_test",
                "permissions": ["search:read"],
            }

    def fake_post(url, **kwargs):
        posted["url"] = url
        posted.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(auth.httpx, "post", fake_post)
    settings = Settings(
        CONVEX_HTTP_URL="https://example.convex.site",
        CONVEX_RECORDING_SECRET="secret",
        RATE_LIMIT_MAX_REQUESTS=7,
        RATE_LIMIT_WINDOW_SECONDS=30,
    )

    context = auth.validate_mechbase_api_key("sk_test", settings=settings)

    assert context.api_key_id == "api_key_test"
    assert posted["json"]["rateLimit"] == {
        "enabled": True,
        "maxRequests": 7,
        "windowSeconds": 30,
    }


def test_validate_api_key_maps_rate_limit_response(monkeypatch) -> None:
    class FakeResponse:
        status_code = 429
        headers = {
            "retry-after": "12",
            "x-ratelimit-limit": "20",
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1710000000000",
        }

        def json(self):
            return {"error": "API key rate limit exceeded."}

    def fake_post(url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(auth.httpx, "post", fake_post)
    settings = Settings(
        CONVEX_HTTP_URL="https://example.convex.site",
        CONVEX_RECORDING_SECRET="secret",
    )

    client = TestClient(app)

    def fake_get_settings():
        return settings

    monkeypatch.setattr(auth, "get_settings", fake_get_settings)
    response = client.get("/auth/validate", headers={"Authorization": "Bearer sk_test"})

    assert response.status_code == 429
    assert response.json() == {"detail": "API key rate limit exceeded."}
    assert response.headers["retry-after"] == "12"
    assert response.headers["x-ratelimit-limit"] == "20"
    assert response.headers["x-ratelimit-remaining"] == "0"
    assert response.headers["x-ratelimit-reset"] == "1710000000000"


def test_search_rejects_key_without_required_permission(monkeypatch) -> None:
    def fake_validate_mechbase_api_key(value, settings=None):
        return ApiKeyContext(
            api_key_id="api_key_test",
            organization_id="workspace_test",
            permissions=("other:read",),
        )

    monkeypatch.setattr(auth, "validate_mechbase_api_key", fake_validate_mechbase_api_key)
    client = TestClient(app)
    response = client.post(
        "/search",
        headers={"Authorization": "Bearer sk_test"},
        json={"query": "shooter"},
    )
    assert response.status_code == 403


def test_search_accepts_valid_key_and_records_usage(monkeypatch) -> None:
    recorded = {}

    def fake_validate_mechbase_api_key(value, settings=None):
        return ApiKeyContext(
            api_key_id="api_key_test",
            organization_id="workspace_test",
            permissions=("search:read",),
        )

    def fake_search(query, **kwargs):
        return SearchResponse(query=query, results=[])

    def fake_record_usage(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(auth, "validate_mechbase_api_key", fake_validate_mechbase_api_key)
    monkeypatch.setattr(main, "search", fake_search)
    monkeypatch.setattr(main, "record_usage", fake_record_usage)

    client = TestClient(app)
    response = client.post(
        "/search",
        headers={"Authorization": "Bearer sk_test"},
        json={"query": "shooter"},
    )

    assert response.status_code == 200
    assert response.json() == {"query": "shooter", "results": []}
    assert recorded["context"].api_key_id == "api_key_test"
    assert recorded["status_code"] == 200
