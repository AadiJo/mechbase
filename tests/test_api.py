from fastapi.testclient import TestClient

import app.api.auth as auth
import app.api.main as main
from app.api.auth import ApiKeyContext
from app.api.main import app
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
