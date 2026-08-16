import asyncio
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.auth as auth
import app.api.main as main
from app.api.auth import ApiKeyContext
from app.api.main import app
from app.rag.config import Settings
from app.rag.ingest import ingestion_lock
from app.rag.models import SearchResponse
from app.rag.store import RagStore


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_api_startup_migrates_and_denies_legacy_control_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    state_dir = tmp_path / "state"
    artifact_dir.mkdir()
    (artifact_dir / "active-generations.json").write_text(
        '{"254":"generation-a"}', encoding="utf-8"
    )
    (artifact_dir / "ingestion-manifest.jsonl").write_text(
        '{"source":"254.pdf"}\n', encoding="utf-8"
    )
    (artifact_dir / ".embedding-cache").mkdir()
    (artifact_dir / ".embedding-cache" / "preview.jpg").write_bytes(b"private")
    nested_metadata = artifact_dir / "sources" / "254" / "generation" / ".complete.json"
    nested_metadata.parent.mkdir(parents=True)
    nested_metadata.write_text("{}", encoding="utf-8")
    settings = Settings(ARTIFACT_DIR=artifact_dir, RAG_STATE_DIR=state_dir)

    main.prepare_control_state(settings)

    store = RagStore(settings, client=SimpleNamespace())
    assert store.active_generations() == {"254": "generation-a"}
    assert not (artifact_dir / "active-generations.json").exists()
    assert not (artifact_dir / "ingestion-manifest.jsonl").exists()
    static_app = FastAPI()
    static_app.mount("/images", main.ArtifactStaticFiles(directory=artifact_dir))
    client = TestClient(static_app)
    assert client.get("/images/active-generations.json").status_code == 404
    assert client.get("/images/ingestion-manifest.jsonl").status_code == 404
    assert client.get("/images/ingestion.lock").status_code == 404
    assert client.get("/images/.embedding-cache/preview.jpg").status_code == 404
    assert client.get("/images/sources/254/generation/.complete.json").status_code == 404


def test_api_startup_waits_for_an_active_legacy_ingestion(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    state_dir = tmp_path / "state"
    settings = Settings(ARTIFACT_DIR=artifact_dir, RAG_STATE_DIR=state_dir)
    writer_started = threading.Event()
    release_writer = threading.Event()
    startup_finished = threading.Event()
    startup_errors: list[BaseException] = []

    def legacy_writer() -> None:
        with ingestion_lock(artifact_dir):
            writer_started.set()
            release_writer.wait(timeout=2)

    def prepare_api() -> None:
        try:
            main.prepare_control_state(settings)
        except BaseException as exc:
            startup_errors.append(exc)
        finally:
            startup_finished.set()

    writer = threading.Thread(target=legacy_writer)
    writer.start()
    assert writer_started.wait(timeout=1)
    startup = threading.Thread(target=prepare_api)
    startup.start()
    assert not startup_finished.wait(timeout=0.05)
    release_writer.set()
    writer.join(timeout=1)
    startup.join(timeout=1)

    assert startup_finished.is_set()
    assert startup_errors == []


def test_compose_services_share_the_configured_private_state_volume() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("- ./rag_state:/app/rag-state") == 2
    assert compose.count("RAG_STATE_DIR: /app/rag-state") == 2


def test_repeatedly_cancelled_startup_awaits_its_bounded_index_worker() -> None:
    async def verify() -> tuple[bool, bool]:
        started = threading.Event()
        release = threading.Event()

        def worker() -> None:
            started.set()
            release.wait(timeout=2)

        task = asyncio.create_task(main._run_blocking_safely(worker))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        still_waiting = not task.done()
        release.set()
        with suppress(asyncio.CancelledError):
            await task
        return still_waiting, task.cancelled()

    assert asyncio.run(verify()) == (True, True)


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
    assert settings.qdrant_timeout_seconds == 10
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
    assert response.json() == {
        "query": "shooter",
        "results": [],
        "coverage": {
            "candidate_pages": 0,
            "candidate_sources": 0,
            "weak_pages_dropped": 0,
            "returned_pages": 0,
            "candidate_window_truncated": False,
        },
        "abstention_reason": None,
    }
    assert recorded["context"].api_key_id == "api_key_test"
    assert recorded["status_code"] == 200


def test_similar_rejects_unbounded_result_windows(monkeypatch) -> None:
    def fake_validate_mechbase_api_key(value, settings=None):
        return ApiKeyContext(
            api_key_id="api_key_test",
            organization_id="workspace_test",
            permissions=("search:read",),
        )

    monkeypatch.setattr(auth, "validate_mechbase_api_key", fake_validate_mechbase_api_key)
    response = TestClient(app).get(
        "/similar",
        params={"result_id": "result", "top_k": 1_000_000},
        headers={"Authorization": "Bearer sk_test"},
    )

    assert response.status_code == 422
