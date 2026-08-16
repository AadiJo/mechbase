import json
from pathlib import Path

import pytest
from PIL import Image

from app.rag.artifacts import source_artifact_root
from app.rag.config import Settings
from app.rag.image_cache import cached_resized_image
from app.rag.ingest import (
    artifact_fingerprint,
    completed_sources,
    ingest_sources,
    ingestion_fingerprint,
    ingestion_lock,
    migrate_legacy_control_state,
    publish_artifact_generation,
    remove_abandoned_artifact_staging,
    remove_artifact_generation,
    with_published_artifacts,
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


def test_artifact_publication_is_content_addressed_and_preserves_history(
    tmp_path: Path,
) -> None:
    root = source_artifact_root(tmp_path, "254-2023")
    root.mkdir(parents=True)
    historical = root / "version-old~fingerprint-old"
    historical.mkdir()
    (historical / "page.png").write_bytes(b"historical")
    first_staging = root / "version-current~ingestion-one"
    first_staging.mkdir()
    first_image = first_staging / "page.png"
    Image.new("RGB", (32, 32), "white").save(first_image)
    canonical_bytes = first_image.read_bytes()
    published_namespace = "version-current~fingerprint-current"

    first_published = publish_artifact_generation(
        tmp_path,
        "254-2023",
        first_staging.name,
        published_namespace,
    )

    published = root / published_namespace
    assert first_published == published_namespace
    assert not first_staging.exists()
    assert (published / "page.png").read_bytes() == canonical_bytes
    assert (published / ".complete.json").is_file()
    assert (historical / "page.png").read_bytes() == b"historical"

    cache_dir = tmp_path / ".embedding-cache"
    embedded = cached_resized_image(published / "page.png", 16, cache_dir)
    assert embedded.is_file()
    assert not (published / ".embed").exists()

    embedded.write_bytes(b"interrupted")
    rebuilt = cached_resized_image(published / "page.png", 16, cache_dir)
    with Image.open(rebuilt) as rebuilt_image:
        rebuilt_image.verify()
        assert rebuilt_image.format == "JPEG"

    duplicate_staging = root / "version-current~ingestion-two"
    duplicate_staging.mkdir()
    (duplicate_staging / "page.png").write_bytes(canonical_bytes)

    duplicate_published = publish_artifact_generation(
        tmp_path,
        "254-2023",
        duplicate_staging.name,
        published_namespace,
    )

    assert duplicate_published == published_namespace
    assert not duplicate_staging.exists()
    assert (published / "page.png").read_bytes() == canonical_bytes
    assert (historical / "page.png").read_bytes() == b"historical"

    (published / ".complete.json").write_text("{}", encoding="utf-8")
    (published / "page.png").write_bytes(b"corrupt")
    repair_staging = root / "version-current~ingestion-three"
    repair_staging.mkdir()
    (repair_staging / "page.png").write_bytes(b"repaired")

    repaired_namespace = publish_artifact_generation(
        tmp_path,
        "254-2023",
        repair_staging.name,
        published_namespace,
    )

    assert not repair_staging.exists()
    assert repaired_namespace != published_namespace
    assert (published / "page.png").read_bytes() == b"corrupt"
    repaired = root / repaired_namespace
    assert (repaired / "page.png").read_bytes() == b"repaired"
    assert (repaired / ".complete.json").is_file()


def test_missing_artifact_staging_is_fatal(tmp_path: Path) -> None:
    root = source_artifact_root(tmp_path, "254-2023")
    root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="staging generation .* is missing"):
        publish_artifact_generation(
            tmp_path,
            "254-2023",
            "version~missing-ingestion",
            "version~fingerprint",
        )


def test_missing_referenced_artifact_never_activates_qdrant_generation(
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

    class PublicationStore:
        def __init__(self) -> None:
            self.active = {source.source_id: "old"}
            self.qdrant_published = False
            self.pointer_activated = False
            self.failed_generation_deleted = False

        def initialize_active_generation(self, _source_id: str) -> None:
            pass

        def active_generations(self) -> dict[str, str]:
            return self.active.copy()

        def upsert(self, docs, text_vectors, image_vectors) -> None:
            assert docs and text_vectors and image_vectors

        def publish_source_generation(self, _source_id: str, _ingestion_id: str) -> None:
            self.qdrant_published = True

        def set_active_generation(self, _source_id: str, _ingestion_id: str) -> None:
            self.pointer_activated = True

        def delete_source_generation(self, _source_id: str, _ingestion_id: str) -> None:
            self.failed_generation_deleted = True

    class FakeEmbedder:
        def embed_texts(self, texts, input_type):
            return [[1.0] for _text in texts]

    def missing_artifact_documents(*_args, **kwargs):
        staging = source_artifact_root(tmp_path, source.source_id) / kwargs["artifact_namespace"]
        return [
            document.model_copy(update={"linked_artifacts": [str(staging / "missing-page.png")]})
        ]

    store = PublicationStore()
    monkeypatch.setattr("app.rag.ingest.extract_documents", missing_artifact_documents)

    with pytest.raises(RuntimeError, match="referenced files missing"):
        ingest_sources(
            [source],
            batch_size=1,
            force=True,
            settings=Settings(ARTIFACT_DIR=tmp_path, EMBEDDING_DIM=1),
            store=store,
            embedder=FakeEmbedder(),
        )

    assert store.qdrant_published is False
    assert store.pointer_activated is False
    assert store.failed_generation_deleted is True


def test_artifact_fingerprint_is_stable_across_ingestions(tmp_path: Path) -> None:
    path = tmp_path / "254-2025.pdf"
    path.write_bytes(b"binder")
    source = parse_source(path)

    first = artifact_fingerprint(source, Settings(RENDER_DPI=150))
    second = artifact_fingerprint(source, Settings(RENDER_DPI=150))
    changed = artifact_fingerprint(source, Settings(RENDER_DPI=200))

    assert first == second
    assert first != changed


def test_document_artifacts_are_rewritten_to_published_namespace(tmp_path: Path) -> None:
    staging = tmp_path / "version~ingestion"
    published = tmp_path / "version~fingerprint"
    document = RagDocument(
        id="result",
        source_id="254-2025",
        source_version="version",
        source_version_id="254-2025@version",
        source_pdf="254-2025.pdf",
        page=1,
        modality="page_image",
        artifact_path=str(staging / "page-001" / "page.png"),
        linked_artifacts=[str(staging / "page-001" / "image.png")],
    )

    rewritten = with_published_artifacts(document, staging, published)

    assert rewritten.artifact_path == str(published / "page-001" / "page.png")
    assert rewritten.linked_artifacts == [str(published / "page-001" / "image.png")]


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
    (first_root / "old").mkdir()
    (prefixed_root / "other-source").mkdir()

    remove_artifact_generation(tmp_path, "254", "old")

    assert not (first_root / "old").exists()
    assert (prefixed_root / "other-source").is_dir()


def test_abandoned_random_staging_is_reclaimed_under_ingestion_lock(tmp_path: Path) -> None:
    root = source_artifact_root(tmp_path, "254")
    root.mkdir(parents=True)
    abandoned = root / f"version~{'a' * 32}"
    published = root / f"version~{'b' * 16}"
    repaired = root / f"version~{'b' * 16}~repair-{'c' * 16}"
    abandoned.mkdir()
    published.mkdir()
    repaired.mkdir()

    with ingestion_lock(tmp_path):
        remove_abandoned_artifact_staging(tmp_path)

    assert not abandoned.exists()
    assert published.is_dir()
    assert repaired.is_dir()


def test_ingestion_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with (
        ingestion_lock(tmp_path),
        pytest.raises(RuntimeError, match="already running"),
        ingestion_lock(tmp_path),
    ):
        pass


def test_legacy_public_control_files_migrate_to_private_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    state_dir = tmp_path / "state"
    artifact_dir.mkdir()
    legacy_manifest = artifact_dir / "ingestion-manifest.jsonl"
    legacy_active = artifact_dir / "active-generations.json"
    legacy_manifest.write_text('{"source":"254.pdf"}\n', encoding="utf-8")
    legacy_active.write_text('{"254":"generation-a"}', encoding="utf-8")
    settings = Settings(ARTIFACT_DIR=artifact_dir, RAG_STATE_DIR=state_dir)

    migrate_legacy_control_state(settings)

    assert (state_dir / legacy_manifest.name).read_text(encoding="utf-8") == (
        '{"source":"254.pdf"}\n'
    )
    assert (state_dir / legacy_active.name).read_text(encoding="utf-8") == (
        '{"254":"generation-a"}'
    )
    assert not legacy_manifest.exists()
    assert not legacy_active.exists()

    legacy_manifest.write_text("stale", encoding="utf-8")
    migrate_legacy_control_state(settings)
    assert not legacy_manifest.exists()
    assert (state_dir / legacy_manifest.name).read_text(encoding="utf-8") == (
        '{"source":"254.pdf"}\n'
    )


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
    settings = Settings(ARTIFACT_DIR=tmp_path, EMBEDDING_DIM=1)
    monkeypatch.setattr("app.rag.ingest.extract_documents", lambda *_args, **_kwargs: [document])

    with pytest.raises(RuntimeError, match="retirement failed"):
        ingest_sources(
            [source],
            batch_size=1,
            force=True,
            settings=settings,
            store=store,
            embedder=FakeEmbedder(),
        )

    assert store.active[source.source_id] != "old"
    assert store.failed_generation_deleted is False
    assert completed_sources(settings.rag_state_dir / "ingestion-manifest.jsonl") == {}


def test_normal_retry_finalizes_a_failed_same_fingerprint_force_refresh(
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
    settings = Settings(ARTIFACT_DIR=tmp_path, EMBEDDING_DIM=1)
    fingerprint = ingestion_fingerprint(source, settings)
    manifest = settings.rag_state_dir / "ingestion-manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "source": source_path.name,
                "ingestion_id": "old",
                "ingestion_fingerprint": fingerprint,
            }
        )
        + "\n",
        encoding="utf-8",
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

    class RetryStore:
        def __init__(self) -> None:
            self.active = {source.source_id: "old"}
            self.upserts = 0
            self.retirements = 0

        def initialize_active_generation(self, _source_id: str) -> None:
            pass

        def active_generations(self) -> dict[str, str]:
            return self.active.copy()

        def upsert(self, docs, text_vectors, image_vectors) -> None:
            assert docs and text_vectors and image_vectors
            self.upserts += 1

        def publish_source_generation(self, _source_id: str, _ingestion_id: str) -> None:
            pass

        def set_active_generation(self, source_id: str, ingestion_id: str) -> None:
            self.active[source_id] = ingestion_id

        def delete_source_generation(self, _source_id: str, _ingestion_id: str) -> None:
            raise AssertionError("an active generation must not be deleted")

        def retire_superseded_source_generations(self, _source_id: str, _ingestion_id: str) -> None:
            self.retirements += 1
            if self.retirements == 1:
                raise RuntimeError("retirement failed")

        def delete_superseded_source_generations(self, _source_id: str, _ingestion_id: str) -> None:
            pass

    class FakeEmbedder:
        def embed_texts(self, texts, input_type):
            return [[1.0] for _text in texts]

    store = RetryStore()
    monkeypatch.setattr("app.rag.ingest.extract_documents", lambda *_args, **_kwargs: [document])

    with pytest.raises(RuntimeError, match="retirement failed"):
        ingest_sources(
            [source],
            batch_size=1,
            force=True,
            settings=settings,
            store=store,
            embedder=FakeEmbedder(),
        )

    first_active = store.active[source.source_id]
    ingest_sources(
        [source],
        batch_size=1,
        force=False,
        settings=settings,
        store=store,
        embedder=FakeEmbedder(),
    )

    assert store.upserts == 2
    assert store.active[source.source_id] != first_active
    assert completed_sources(manifest) == {source_path.name: fingerprint}
    assert (
        json.loads(manifest.read_text(encoding="utf-8").splitlines()[-1])["ingestion_id"]
        == (store.active[source.source_id])
    )


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
