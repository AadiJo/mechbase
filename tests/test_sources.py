import json
from pathlib import Path

import pytest

from app.rag.artifacts import source_artifact_root
from app.rag.config import Settings
from app.rag.ingest import (
    completed_sources,
    ingest_sources,
    ingestion_fingerprint,
    ingestion_lock,
    remove_artifact_generation,
    remove_superseded_artifacts,
)
from app.rag.models import RagDocument, SourceDoc
from app.rag.sources import iter_pdfs, parse_source


def test_parse_team_year_filename(tmp_path: Path) -> None:
    path = tmp_path / "254-2025.pdf"
    path.write_bytes(b"binder-version-one")
    source = parse_source(path)
    assert source.team == "254"
    assert source.year == 2025
    assert source.source_id == "254-2025"
    assert source.source_version_id.startswith("254-2025@")


def test_parse_multi_part_filename(tmp_path: Path) -> None:
    path = tmp_path / "4607-2-2024.pdf"
    path.write_bytes(b"binder")
    source = parse_source(path)
    assert source.team == "4607"
    assert source.year == 2024


def test_source_version_changes_with_pdf_content(tmp_path: Path) -> None:
    path = tmp_path / "254-2025.pdf"
    path.write_bytes(b"first")
    first = parse_source(path)
    path.write_bytes(b"second")
    second = parse_source(path)

    assert first.source_id == second.source_id
    assert first.source_version != second.source_version
    assert first.source_version_id != second.source_version_id


def test_iter_pdfs_loads_original_source_urls(tmp_path: Path) -> None:
    pdf = tmp_path / "254-2025.pdf"
    pdf.touch()
    (tmp_path / "sources.json").write_text(
        json.dumps({pdf.name: "https://example.com/254-2025.pdf"}),
        encoding="utf-8",
    )

    sources = iter_pdfs(tmp_path)

    assert sources[0].source_url == "https://example.com/254-2025.pdf"


def test_completed_sources_tracks_latest_ingestion_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "ingestion-manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"source": "254-2025.pdf", "ingestion_fingerprint": "old"}),
                json.dumps({"source": "254-2025.pdf", "ingestion_fingerprint": "new"}),
            ]
        ),
        encoding="utf-8",
    )

    assert completed_sources(manifest) == {"254-2025.pdf": "new"}


def test_ingestion_fingerprint_tracks_provenance_and_embedding_config(tmp_path: Path) -> None:
    path = tmp_path / "254-2025.pdf"
    path.write_bytes(b"binder")
    source = parse_source(path)

    baseline = ingestion_fingerprint(source, Settings())
    changed_source = source.model_copy(update={"source_url": "https://example.com/binder.pdf"})
    changed_settings = Settings(CHUNK_TARGET_CHARS=900, TEXT_MODEL="replacement-model")

    assert ingestion_fingerprint(changed_source, Settings()) != baseline
    assert ingestion_fingerprint(source, changed_settings) != baseline


def test_remove_superseded_artifacts_only_removes_matching_source_generations(
    tmp_path: Path,
) -> None:
    current = "254-2023@version#current"
    root = source_artifact_root(tmp_path, "254-2023")
    root.mkdir(parents=True)
    for name in ["254-2023", "254-2023@old#old", current, "254-20230@other#other"]:
        (root / name).mkdir()

    remove_superseded_artifacts(tmp_path, "254-2023", current)

    assert not (root / "254-2023").exists()
    assert not (root / "254-2023@old#old").exists()
    assert (root / current).is_dir()
    assert not (root / "254-20230@other#other").exists()


def test_remove_artifact_generation_removes_only_the_exact_namespace(tmp_path: Path) -> None:
    root = source_artifact_root(tmp_path, "254-2023")
    root.mkdir(parents=True)
    target = root / "254-2023@version#failed"
    neighbor = root / "254-2023@version#complete"
    target.mkdir()
    neighbor.mkdir()

    remove_artifact_generation(tmp_path, "254-2023", target.name)

    assert not target.exists()
    assert neighbor.is_dir()


def test_artifact_cleanup_cannot_cross_source_id_prefixes(tmp_path: Path) -> None:
    first_root = source_artifact_root(tmp_path, "254")
    prefixed_root = source_artifact_root(tmp_path, "254@prototype")
    first_root.mkdir(parents=True)
    prefixed_root.mkdir(parents=True)
    (first_root / "current").mkdir()
    (first_root / "old").mkdir()
    (prefixed_root / "other-source").mkdir()

    remove_superseded_artifacts(tmp_path, "254", "current")

    assert (first_root / "current").is_dir()
    assert not (first_root / "old").exists()
    assert (prefixed_root / "other-source").is_dir()


def test_ingestion_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with (
        ingestion_lock(tmp_path),
        pytest.raises(RuntimeError, match="already running"),
        ingestion_lock(tmp_path),
    ):
        pass


def test_committed_generation_stays_authoritative_when_retirement_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "254-2025.pdf"
    source_path.write_bytes(b"binder")
    source = SourceDoc(
        path=source_path,
        team="254",
        year=2025,
        source_id="254-2025",
        source_version="content",
        source_version_id="254-2025@content",
    )
    document = RagDocument(
        id="result",
        storage_id="result@new",
        source_id=source.source_id,
        source_version=source.source_version,
        source_version_id=source.source_version_id,
        source_pdf=source_path.name,
        team="254",
        year=2025,
        page=1,
        modality="text",
        text="intake",
        is_staged=True,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.active: dict[str, str] = {}
            self.failed_generation_deleted = False

        def initialize_active_generation(self, source_id: str) -> None:
            self.active[source_id] = "old"

        def upsert(self, docs, text_vectors, image_vectors) -> None:
            assert docs and text_vectors and image_vectors

        def publish_source_generation(self, source_id: str, ingestion_id: str) -> None:
            pass

        def set_active_generation(self, source_id: str, ingestion_id: str) -> None:
            self.active[source_id] = ingestion_id

        def delete_source_generation(self, source_id: str, ingestion_id: str) -> None:
            self.failed_generation_deleted = True

        def retire_superseded_source_generations(self, source_id: str, ingestion_id: str) -> None:
            pass

        def delete_superseded_source_generations(self, source_id: str, ingestion_id: str) -> None:
            raise RuntimeError("retirement failed")

    class FakeEmbedder:
        def embed_texts(self, texts, input_type):
            return [[1.0] for _text in texts]

    store = FakeStore()
    monkeypatch.setattr("app.rag.ingest.extract_documents", lambda *_args, **_kwargs: [document])

    with pytest.raises(RuntimeError, match="retirement failed"):
        ingest_sources(
            [source],
            batch_size=1,
            force=True,
            settings=Settings(ARTIFACT_DIR=tmp_path, EMBEDDING_DIM=1),
            store=store,
            embedder=FakeEmbedder(),
        )

    assert store.active[source.source_id] != "old"
    assert store.failed_generation_deleted is False
    assert completed_sources(tmp_path / "ingestion-manifest.jsonl") == {}


def test_generation_survives_an_error_after_the_active_pointer_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "254-2025.pdf"
    source_path.write_bytes(b"binder")
    source = SourceDoc(
        path=source_path,
        team="254",
        year=2025,
        source_id="254-2025",
        source_version="content",
        source_version_id="254-2025@content",
    )
    document = RagDocument(
        id="result",
        storage_id="result@new",
        source_id=source.source_id,
        source_version=source.source_version,
        source_version_id=source.source_version_id,
        source_pdf=source_path.name,
        team="254",
        year=2025,
        page=1,
        modality="text",
        text="intake",
        is_staged=True,
    )

    class RenameFailureStore:
        def __init__(self) -> None:
            self.active = {source.source_id: "old"}
            self.failed_generation_deleted = False

        def initialize_active_generation(self, _source_id: str) -> None:
            pass

        def active_generations(self) -> dict[str, str]:
            return self.active.copy()

        def upsert(self, docs, text_vectors, image_vectors) -> None:
            assert docs and text_vectors and image_vectors

        def publish_source_generation(self, _source_id: str, _ingestion_id: str) -> None:
            pass

        def set_active_generation(self, source_id: str, ingestion_id: str) -> None:
            self.active[source_id] = ingestion_id
            raise OSError("directory fsync failed")

        def delete_source_generation(self, _source_id: str, _ingestion_id: str) -> None:
            self.failed_generation_deleted = True

    class FakeEmbedder:
        def embed_texts(self, texts, input_type):
            return [[1.0] for _text in texts]

    store = RenameFailureStore()
    monkeypatch.setattr("app.rag.ingest.extract_documents", lambda *_args, **_kwargs: [document])

    with pytest.raises(OSError, match="directory fsync failed"):
        ingest_sources(
            [source],
            batch_size=1,
            force=True,
            settings=Settings(ARTIFACT_DIR=tmp_path, EMBEDDING_DIM=1),
            store=store,
            embedder=FakeEmbedder(),
        )

    assert store.active[source.source_id] != "old"
    assert store.failed_generation_deleted is False


def test_iter_pdfs_rejects_non_http_source_urls(tmp_path: Path) -> None:
    pdf = tmp_path / "254-2025.pdf"
    pdf.touch()
    (tmp_path / "sources.json").write_text(
        json.dumps({pdf.name: "javascript:alert(1)"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        iter_pdfs(tmp_path)
