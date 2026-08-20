from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.rag.config import Settings
from app.rag.models import (
    PageContextResponse,
    RagDocument,
    SearchCoverage,
    SearchRequest,
    SimilarPagesResponse,
)
from app.rag.pdf import _document_id
from app.rag.search import _expansion_years, search_source_catalog
from app.rag.store import IMAGE_VECTOR, TEXT_VECTOR, RagStore, _build_filter


def test_build_filter_uses_exact_multi_value_metadata_filters() -> None:
    qfilter = _build_filter(
        SearchRequest(
            query="cone intake",
            team_numbers=["254", "4414"],
            years=[2023, 2024],
            source_ids=["254-2023", "4414-2024"],
        )
    )

    assert qfilter is not None
    conditions = {condition.key: condition for condition in qfilter.must or []}
    assert conditions["team"].match.any == ["254", "4414"]
    assert conditions["year"].match.any == [2023, 2024]
    assert conditions["source_id"].match.any == ["254-2023", "4414-2024"]
    assert qfilter.must_not[0].key == "is_staged"
    assert qfilter.must_not[0].match.value is True


def test_public_document_ids_preserve_exact_source_identity() -> None:
    spaced = _document_id("foo bar@same-content", 1, "text", 0)
    underscored = _document_id("foo_bar@same-content", 1, "text", 0)

    assert spaced != underscored
    assert spaced == _document_id("foo bar@same-content", 1, "text", 0)


def test_build_filter_unions_legacy_and_multi_value_filters() -> None:
    qfilter = _build_filter(
        SearchRequest(
            query="shooter",
            team="254",
            team_numbers=["4414"],
            year=2023,
            years=[2024],
            source="254-2023.pdf",
        )
    )

    assert qfilter is not None
    conditions = qfilter.must or []
    team_condition = next(condition for condition in conditions if condition.key == "team")
    year_condition = next(condition for condition in conditions if condition.key == "year")
    source_condition = next(condition for condition in conditions if condition.key == "source_pdf")
    assert team_condition.match.any == ["254", "4414"]
    assert year_condition.match.any == [2023, 2024]
    assert source_condition.match.value == "254-2023.pdf"


def test_query_expansion_uses_the_same_legacy_and_plural_year_union() -> None:
    request = SearchRequest(
        query="multi ball",
        year=2020,
        years=[2024, 2020],
    )

    assert _expansion_years(request) == [2020, 2024]


def test_source_catalog_search_embeds_once_and_applies_every_exact_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingEmbedder:
        def __init__(self) -> None:
            self.text_calls = 0
            self.multimodal_calls = 0

        def embed_texts(self, texts, input_type):
            self.text_calls += 1
            return [[0.1]]

        def embed_multimodal(self, texts, images, input_type):
            self.multimodal_calls += 1
            return [[0.2]]

    class RecordingStore:
        def __init__(self) -> None:
            self.source_ids: list[str] | None = None

        def search(
            self,
            request,
            text_vector,
            image_vector,
            expanded_query,
            *,
            source_ids=None,
        ):
            self.source_ids = source_ids
            return [], SearchCoverage()

    embedder = RecordingEmbedder()
    store = RecordingStore()
    monkeypatch.setattr("app.rag.search.VoyageEmbedder", lambda _settings: embedder)
    source_ids = [f"254-archive-{index}" for index in range(22)]

    search_source_catalog(
        "archive intake",
        8,
        Settings(),
        team_numbers=["254"],
        years=[],
        source_ids=source_ids,
        store=store,  # type: ignore[arg-type]
    )

    assert embedder.text_calls == 1
    assert embedder.multimodal_calls == 1
    assert store.source_ids == source_ids


def test_active_generation_filter_has_constant_condition_count() -> None:
    active = {f"source-{index}": f"ingestion-{index}" for index in range(1000)}

    qfilter = _build_filter(SearchRequest(query="intake"), active)

    generation_filter = qfilter.must[-1]
    assert isinstance(generation_filter, models.Filter)
    assert len(generation_filter.should) == 2
    assert generation_filter.should[0].match.any == list(active.values())


class ExistingCollectionClient:
    def __init__(self) -> None:
        self.created_indexes: list[tuple[str, object]] = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="frc_mechanisms")])

    def get_collection(self, _name: str):
        return SimpleNamespace(payload_schema={"team": {}})

    def create_payload_index(self, **kwargs):
        self.created_indexes.append((kwargs["field_name"], kwargs["field_schema"]))


def test_ensure_collection_adds_missing_metadata_indexes() -> None:
    client = ExistingCollectionClient()
    store = RagStore(Settings(), client=client)

    store.ensure_collection()

    created_fields = {field for field, _schema in client.created_indexes}
    assert created_fields == {
        "id",
        "year",
        "source_id",
        "source_version",
        "source_version_id",
        "ingestion_id",
        "is_staged",
        "source_pdf",
        "page",
        "modality",
        "artifact_path",
        "linked_artifacts",
    }


class MissingCollectionClient:
    def __init__(self) -> None:
        self.created = False

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def create_collection(self, **_kwargs):
        self.created = True


def test_payload_index_migration_does_not_create_a_missing_collection() -> None:
    client = MissingCollectionClient()
    store = RagStore(Settings(), client=client)

    assert store.ensure_payload_indexes() is False
    assert client.created is False


class RevisionClient:
    def __init__(self) -> None:
        self.points_count = 10
        self.indexed_vectors_count = 1

    def get_collection(self, _name: str):
        return SimpleNamespace(
            points_count=self.points_count,
            indexed_vectors_count=self.indexed_vectors_count,
        )


def test_corpus_revision_ignores_optimizer_progress_and_tracks_generation_commit(
    tmp_path: Path,
) -> None:
    client = RevisionClient()
    store = RagStore(Settings(ARTIFACT_DIR=tmp_path), client=client)
    baseline = store.corpus_revision()
    client.points_count = 99
    client.indexed_vectors_count = 9

    assert store.corpus_revision() == baseline

    store.set_active_generation("254-2023", "generation-new")

    assert store.corpus_revision() != baseline


class VersionedPointClient:
    def __init__(self) -> None:
        self.payloads: dict[str, dict] = {}

    def upsert(self, **kwargs):
        for point in kwargs["points"]:
            self.payloads[str(point.id)] = point.payload

    def delete(self, **kwargs):
        qfilter = kwargs["points_selector"].filter
        source_id = qfilter.must[0].match.value
        exact_ingestion = qfilter.must[1].match.value if len(qfilter.must) > 1 else None
        current_ingestion = qfilter.must_not[0].match.value if qfilter.must_not else None
        self.payloads = {
            point_id: payload
            for point_id, payload in self.payloads.items()
            if payload["source_id"] != source_id
            or (
                payload.get("ingestion_id") == current_ingestion
                if current_ingestion
                else payload.get("ingestion_id") != exact_ingestion
            )
        }

    def set_payload(self, **kwargs):
        qfilter = kwargs["points"]
        conditions = {condition.key: condition.match.value for condition in qfilter.must}
        for payload in self.payloads.values():
            if (
                payload["source_id"] == conditions["source_id"]
                and payload.get("ingestion_id") == conditions["ingestion_id"]
            ):
                payload.update(kwargs["payload"])


def _versioned_doc(
    version: str,
    page: int,
    *,
    ingestion_id: str | None = None,
    text: str = "intake",
    ingested_at: str | None = None,
) -> RagDocument:
    version_id = f"254-2023@{version}"
    namespace = f"{version_id}#{ingestion_id}" if ingestion_id else version_id
    return RagDocument(
        id=f"{version_id}_{page}_text_0",
        storage_id=f"{namespace}_{page}_text_0",
        source_id="254-2023",
        source_version=version,
        source_version_id=version_id,
        ingestion_id=ingestion_id,
        source_pdf="254-2023.pdf",
        team="254",
        year=2023,
        page=page,
        modality="text",
        text=text,
        ingested_at=ingested_at,
    )


def test_revised_source_removes_superseded_pages_after_new_version_is_ready() -> None:
    client = VersionedPointClient()
    store = RagStore(Settings(EMBEDDING_DIM=1), client=client)
    old_docs = [
        _versioned_doc("same", 1, ingestion_id="old"),
        _versioned_doc("same", 2, ingestion_id="old"),
    ]
    new_docs = [_versioned_doc("same", 1, ingestion_id="new")]

    store.upsert(old_docs, [[0.0], [0.0]], [[0.0], [0.0]])
    store.upsert(new_docs, [[0.0]], [[0.0]])
    store.delete_superseded_source_generations("254-2023", "new")

    assert len(client.payloads) == 1
    remaining = next(iter(client.payloads.values()))
    assert remaining["source_version"] == "same"
    assert remaining["ingestion_id"] == "new"
    assert remaining["page"] == 1


def test_failed_generation_cleanup_preserves_previous_generation() -> None:
    client = VersionedPointClient()
    store = RagStore(Settings(EMBEDDING_DIM=1), client=client)
    old_doc = _versioned_doc("same", 1, ingestion_id="old")
    failed_doc = _versioned_doc("same", 2, ingestion_id="failed")
    store.upsert([old_doc, failed_doc], [[0.0], [0.0]], [[0.0], [0.0]])

    store.delete_source_generation("254-2023", "failed")

    assert len(client.payloads) == 1
    remaining = next(iter(client.payloads.values()))
    assert remaining["ingestion_id"] == "old"


def test_generation_is_published_only_by_an_explicit_commit() -> None:
    client = VersionedPointClient()
    store = RagStore(Settings(EMBEDDING_DIM=1), client=client)
    staged = _versioned_doc("same", 1, ingestion_id="new").model_copy(update={"is_staged": True})
    store.upsert([staged], [[1.0]], [[1.0]])

    store.publish_source_generation("254-2023", "new")

    published = next(iter(client.payloads.values()))
    assert published["is_staged"] is False


def test_staged_generation_is_hidden_from_source_and_page_reads(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    vector_params = models.VectorParams(size=1, distance=models.Distance.COSINE)
    client.create_collection(
        collection_name="frc_mechanisms",
        vectors_config={TEXT_VECTOR: vector_params, IMAGE_VECTOR: vector_params},
    )
    store = RagStore(Settings(EMBEDDING_DIM=1, ARTIFACT_DIR=tmp_path), client=client)
    active = _versioned_doc(
        "same",
        1,
        ingestion_id="active",
        ingested_at="2026-08-15T12:00:00+00:00",
    )
    staged = _versioned_doc(
        "same",
        2,
        ingestion_id="staged",
        ingested_at="2026-08-16T12:00:00+00:00",
    ).model_copy(update={"is_staged": True})
    store.upsert([active, staged], [[1.0], [1.0]], [[1.0], [1.0]])
    store.initialize_active_generation("254-2023")

    summaries = store.list_sources().sources

    assert store.active_generations()["254-2023"] == "active"
    assert [summary.pages for summary in summaries] == [[1]]
    assert store.page_context("254-2023.pdf", 2) is None

    store.publish_source_generation("254-2023", "staged")

    assert [summary.pages for summary in store.list_sources().sources] == [[1]]

    store.set_active_generation("254-2023", "staged")

    assert [summary.pages for summary in store.list_sources().sources] == [[2]]
    assert store.page_context("254-2023.pdf", 1) is None
    assert store.page_context("254-2023.pdf", 2) is not None
    client.close()


def test_search_filter_excludes_superseded_published_generations(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    vector_params = models.VectorParams(size=1, distance=models.Distance.COSINE)
    client.create_collection(
        collection_name="frc_mechanisms",
        vectors_config={TEXT_VECTOR: vector_params, IMAGE_VECTOR: vector_params},
    )
    store = RagStore(Settings(EMBEDDING_DIM=1, ARTIFACT_DIR=tmp_path), client=client)
    old = _versioned_doc("same", 1, ingestion_id="old", text="old intake")
    current = _versioned_doc("same", 2, ingestion_id="current", text="current intake")
    store.upsert([old, current], [[1.0], [1.0]], [[1.0], [1.0]])
    store.set_active_generation("254-2023", "current")

    results, _coverage = store.search(
        SearchRequest(query="intake", top_k=1),
        [1.0],
        [1.0],
        "intake",
    )

    assert [(result.ingestion_id, result.page) for result in results] == [("current", 2)]
    client.close()


def test_page_citations_are_explicitly_current_source_links(tmp_path: Path) -> None:
    store = RagStore(Settings(ARTIFACT_DIR=tmp_path), client=SimpleNamespace())
    old_payload = _versioned_doc("old", 1, ingestion_id="old").model_dump()
    new_payload = _versioned_doc("new", 1, ingestion_id="new").model_dump()

    old_result = store._search_result_from_payload(old_payload, 0.9)
    new_result = store._search_result_from_payload(new_payload, 0.9)

    assert old_result.page_context_url == new_result.page_context_url
    assert old_result.page_text_url == new_result.page_text_url
    assert old_result.page_context_url == "/pages/254-2023.pdf/1"
    assert "source_version_id" not in old_result.page_context_url
    assert "ingestion_id" not in old_result.page_context_url


def test_read_retries_when_active_generation_changes_mid_query(tmp_path: Path) -> None:
    store = RagStore(Settings(ARTIFACT_DIR=tmp_path), client=SimpleNamespace())
    snapshots = iter(
        [
            {"254-2023": "old"},
            {"254-2023": "new"},
            {"254-2023": "new"},
            {"254-2023": "new"},
        ]
    )
    store.active_generations = lambda: next(snapshots)  # type: ignore[method-assign]
    reads: list[dict[str, str]] = []

    value, active = store._read_with_active_snapshot(lambda snapshot: reads.append(snapshot) or 1)

    assert value == 1
    assert active == {"254-2023": "new"}
    assert reads == [{"254-2023": "old"}, {"254-2023": "new"}]


class SnapshotBoundFetchStore(RagStore):
    def __init__(
        self,
        snapshots: list[dict[str, str]],
        *,
        missing_generations: set[str] | None = None,
    ) -> None:
        super().__init__(Settings(), client=SimpleNamespace())
        self._snapshots = iter(snapshots)
        self.missing_generations = missing_generations or set()
        self.payload_snapshots: list[dict[str, str]] = []
        self.context_snapshots: list[dict[str, str]] = []

    def active_generations(self) -> dict[str, str]:
        return next(self._snapshots)

    def _payload_and_vectors_for_result_id_active(
        self,
        result_id: str,
        active_generations: dict[str, str],
    ) -> tuple[dict | None, dict | None]:
        self.payload_snapshots.append(active_generations)
        ingestion_id = active_generations["254-2023"]
        if ingestion_id in self.missing_generations:
            return None, None
        payload = _versioned_doc(
            "same",
            2,
            ingestion_id=ingestion_id,
            text=f"{ingestion_id} intake",
        ).model_dump()
        payload["id"] = result_id
        return payload, None

    def _page_contexts_for_active(
        self,
        source_pdf: str,
        pages: list[int] | None,
        source_version_id: str | None,
        ingestion_id: str | None,
        active_generations: dict[str, str],
    ) -> list[PageContextResponse]:
        self.context_snapshots.append(active_generations)
        assert source_pdf == "254-2023.pdf"
        assert source_version_id == "254-2023@same"
        assert ingestion_id == active_generations["254-2023"]
        return [
            PageContextResponse(
                source_id="254-2023",
                source_version="same",
                source_version_id=source_version_id,
                ingestion_id=ingestion_id,
                source_pdf=source_pdf,
                team="254",
                year=2023,
                page=page,
                text=f"{ingestion_id} page {page}",
            )
            for page in pages or []
        ]


def test_fetch_contexts_retries_the_whole_source_read_after_activation() -> None:
    old = {"254-2023": "old", "4414-2023": "steady"}
    new = {"254-2023": "new", "4414-2023": "steady"}
    store = SnapshotBoundFetchStore([old, new, new, new])

    response = store.fetch_contexts("result", adjacent_pages=1)

    assert response is not None
    assert response.context.ingestion_id == "new"
    assert [context.ingestion_id for context in response.adjacent_contexts] == ["new", "new"]
    assert store.payload_snapshots == [old, new]
    assert store.context_snapshots == [old, new]


def test_fetch_contexts_ignores_unrelated_source_activation() -> None:
    before = {"254-2023": "steady", "4414-2023": "old"}
    after = {"254-2023": "steady", "4414-2023": "new"}
    store = SnapshotBoundFetchStore([before, after])

    response = store.fetch_contexts("result", adjacent_pages=0)

    assert response is not None
    assert response.context.ingestion_id == "steady"
    assert store.payload_snapshots == [before]
    assert store.context_snapshots == [before]


def test_fetch_contexts_retries_a_missing_id_when_the_active_map_changes() -> None:
    old = {"254-2023": "old"}
    new = {"254-2023": "new"}
    store = SnapshotBoundFetchStore(
        [old, new, new, new],
        missing_generations={"old"},
    )

    response = store.fetch_contexts("new-result", adjacent_pages=0)

    assert response is not None
    assert response.context.result_id == "new-result"
    assert response.context.ingestion_id == "new"
    assert store.payload_snapshots == [old, new]
    assert store.context_snapshots == [new]


def test_fetch_contexts_reports_repeated_activation_while_id_is_missing() -> None:
    first = {"254-2023": "first"}
    second = {"254-2023": "second"}
    third = {"254-2023": "third"}
    fourth = {"254-2023": "fourth"}
    fifth = {"254-2023": "fifth"}
    sixth = {"254-2023": "sixth"}
    store = SnapshotBoundFetchStore(
        [first, second, third, fourth, fifth, sixth],
        missing_generations={"first", "third", "fifth"},
    )

    with pytest.raises(RuntimeError, match="while fetching result 'missing'"):
        store.fetch_contexts("missing", adjacent_pages=0)

    assert store.payload_snapshots == [first, third, fifth]
    assert store.context_snapshots == []


class SnapshotBoundBrowseStore(RagStore):
    def __init__(self, snapshots: list[dict[str, str]]) -> None:
        super().__init__(Settings(), client=SimpleNamespace())
        self._snapshots = iter(snapshots)
        self.current_snapshot: dict[str, str] = {}
        self.summary_snapshots: list[dict[str, str]] = []
        self.context_snapshots: list[dict[str, str]] = []

    def active_generations(self) -> dict[str, str]:
        self.current_snapshot = next(self._snapshots)
        return self.current_snapshot

    def _iter_source_payloads(self, qfilter: models.Filter | None):
        assert qfilter is not None
        snapshot = dict(self.current_snapshot)
        self.summary_snapshots.append(snapshot)
        ingestion_id = snapshot["254-2023"]
        for page in (1, 2):
            yield _versioned_doc(
                "same",
                page,
                ingestion_id=ingestion_id,
                text=f"{ingestion_id} page {page}",
            ).model_dump()

    def _page_contexts_for_active(
        self,
        source_pdf: str,
        pages: list[int] | None,
        source_version_id: str | None,
        ingestion_id: str | None,
        active_generations: dict[str, str],
    ) -> list[PageContextResponse]:
        self.context_snapshots.append(active_generations)
        return [
            PageContextResponse(
                source_id="254-2023",
                source_version="same",
                source_version_id=source_version_id,
                ingestion_id=ingestion_id,
                source_pdf=source_pdf,
                team="254",
                year=2023,
                page=page,
                text=f"{ingestion_id} page {page}",
            )
            for page in pages or []
        ]


def test_browse_contexts_retries_summary_and_pages_under_one_source_snapshot() -> None:
    old = {"254-2023": "old", "4414-2023": "steady"}
    new = {"254-2023": "new", "4414-2023": "steady"}
    store = SnapshotBoundBrowseStore([old, new, new, new])

    response = store.browse_contexts(
        "254-2023",
        start_page=1,
        end_page=2,
        resume_page=None,
        scan_limit=10,
    )

    assert response is not None
    assert response.source.ingestion_id == "new"
    assert [context.ingestion_id for context in response.contexts] == ["new", "new"]
    assert store.summary_snapshots == [old, new]
    assert store.context_snapshots == [old, new]


def test_browse_contexts_reuses_the_generation_pinned_source_summary() -> None:
    active = {"254-2023": "steady"}
    store = SnapshotBoundBrowseStore([active, active, active, active])

    first = store.browse_contexts(
        "254-2023",
        start_page=None,
        end_page=None,
        resume_page=None,
        scan_limit=1,
    )
    second = store.browse_contexts(
        "254-2023",
        start_page=None,
        end_page=None,
        resume_page=2,
        scan_limit=1,
    )

    assert first is not None
    assert second is not None
    assert first.requested_pages == [1]
    assert second.requested_pages == [2]
    assert store.summary_snapshots == [active]
    assert store.context_snapshots == [active, active]


def test_source_summaries_do_not_mix_versions() -> None:
    store = RagStore(Settings(), client=SimpleNamespace())

    summaries = store._summarize_sources(
        [
            _versioned_doc("old", 1).model_dump(),
            _versioned_doc("new", 1).model_dump(),
        ]
    )

    assert {summary.source_version_id for summary in summaries} == {
        "254-2023@old",
        "254-2023@new",
    }


def test_source_summaries_do_not_mix_generations_of_the_same_version() -> None:
    store = RagStore(Settings(), client=SimpleNamespace())

    summaries = store._summarize_sources(
        [
            _versioned_doc("same", 1, ingestion_id="old").model_dump(),
            _versioned_doc("same", 1, ingestion_id="new").model_dump(),
        ]
    )

    assert {summary.ingestion_id for summary in summaries} == {"old", "new"}


def test_source_summary_coalesces_provenance_from_all_version_payloads() -> None:
    store = RagStore(Settings(), client=SimpleNamespace())
    first = _versioned_doc("new", 1).model_dump()
    second = _versioned_doc("new", 2).model_dump()
    second["source_url"] = "https://example.com/254-2023.pdf"
    second["ingested_at"] = "2026-08-16T12:00:00+00:00"

    summary = store._summarize_sources([first, second])[0]

    assert summary.source_url == "https://example.com/254-2023.pdf"
    assert summary.ingested_at == "2026-08-16T12:00:00+00:00"


class VersionedPageClient:
    def __init__(self) -> None:
        self.scroll_filter = None

    def scroll(self, **kwargs):
        self.scroll_filter = kwargs["scroll_filter"]
        return (
            [
                SimpleNamespace(
                    payload={
                        **_versioned_doc("new", 1, ingestion_id="generation-new").model_dump(),
                        "id": "result-new",
                    }
                )
            ],
            None,
        )


def test_page_context_is_scoped_to_the_result_generation() -> None:
    client = VersionedPageClient()
    store = RagStore(Settings(), client=client)

    context = store.page_context("254-2023.pdf", 1, "254-2023@new", "generation-new")

    assert context is not None
    conditions = {condition.key: condition for condition in client.scroll_filter.must}
    assert conditions["source_version_id"].match.value == "254-2023@new"
    assert conditions["ingestion_id"].match.value == "generation-new"
    assert client.scroll_filter.must_not[0].key == "is_staged"
    assert context.source_version_id == "254-2023@new"


class BatchPageClient:
    def __init__(self) -> None:
        self.calls = 0
        self.scroll_filter = None

    def scroll(self, **kwargs):
        self.calls += 1
        self.scroll_filter = kwargs["scroll_filter"]
        return (
            [
                SimpleNamespace(
                    payload={
                        **_versioned_doc(
                            "new",
                            page,
                            ingestion_id="generation-new",
                            text=f"page {page}",
                        ).model_dump(),
                        "id": f"result-{page}",
                    }
                )
                for page in (2, 1)
            ],
            None,
        )


def test_page_contexts_loads_multiple_pages_in_one_snapshot_read() -> None:
    client = BatchPageClient()
    store = RagStore(Settings(), client=client)

    contexts = store.page_contexts(
        "254-2023.pdf",
        [2, 1],
        "254-2023@new",
        "generation-new",
    )

    assert client.calls == 1
    assert [context.page for context in contexts] == [1, 2]
    assert [context.text for context in contexts] == ["page 1", "page 2"]
    conditions = {condition.key: condition for condition in client.scroll_filter.must}
    assert conditions["page"].match.any == [2, 1]


class BatchImageContextClient:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.calls = 0
        self.filters = []

    def scroll(self, **kwargs):
        self.calls += 1
        scroll_filter = kwargs["scroll_filter"]
        self.filters.append(scroll_filter)
        if any(getattr(condition, "key", None) == "id" for condition in scroll_filter.must):
            return (
                [
                    SimpleNamespace(
                        payload={
                            **_versioned_doc(
                                "new",
                                page,
                                ingestion_id="generation-new",
                                text=f"page {page}",
                            ).model_dump(),
                            "id": f"result-{page}",
                        }
                    )
                    for page in (1, 2)
                ],
                None,
            )

        payloads = []
        for page in (1, 2):
            text_payload = {
                **_versioned_doc(
                    "new",
                    page,
                    ingestion_id="generation-new",
                    text=f"page {page}",
                ).model_dump(),
                "id": f"result-{page}",
            }
            page_payload = {
                **text_payload,
                "id": f"page-result-{page}",
                "modality": "page_image",
                "artifact_path": str(
                    self.artifact_dir / f"generation-new/page-{page:03d}/page.png"
                ),
            }
            payloads.extend(
                [
                    SimpleNamespace(payload=text_payload),
                    SimpleNamespace(payload=page_payload),
                ]
            )
        return payloads, None


def test_image_contexts_batches_result_and_page_lookups(tmp_path: Path) -> None:
    client = BatchImageContextClient(tmp_path)
    store = RagStore(Settings(ARTIFACT_DIR=tmp_path), client=client)

    contexts = store.image_contexts(["result-1", "result-2"])

    assert client.calls == 2
    assert len(client.filters[1].should) == 2
    assert list(contexts) == ["result-1", "result-2"]
    assert contexts["result-1"].page_image_url == ("/images/generation-new/page-001/page.png")
    assert contexts["result-2"].text == "page 2"


class MixedGenerationPageClient:
    def scroll(self, **_kwargs):
        old = _versioned_doc(
            "same",
            1,
            ingestion_id="generation-old",
            text="old text",
            ingested_at="2026-08-15T12:00:00+00:00",
        ).model_dump()
        new = _versioned_doc(
            "same",
            1,
            ingestion_id="generation-new",
            text="new text",
            ingested_at="2026-08-16T12:00:00+00:00",
        ).model_dump()
        return ([SimpleNamespace(payload=old), SimpleNamespace(payload=new)], None)


def test_versionless_page_context_uses_latest_complete_generation() -> None:
    store = RagStore(Settings(), client=MixedGenerationPageClient())

    context = store.page_context("254-2023.pdf", 1)

    assert context is not None
    assert context.ingestion_id == "generation-new"
    assert context.text == "new text"


class StableResultLookupClient:
    def __init__(self) -> None:
        self.scroll_filter = None

    def scroll(self, **kwargs):
        self.scroll_filter = kwargs["scroll_filter"]
        old = _versioned_doc(
            "same",
            1,
            ingestion_id="generation-old",
            ingested_at="2026-08-15T12:00:00+00:00",
        ).model_dump()
        new = _versioned_doc(
            "same",
            1,
            ingestion_id="generation-new",
            ingested_at="2026-08-16T12:00:00+00:00",
        ).model_dump()
        return (
            [
                SimpleNamespace(payload=old, vector={TEXT_VECTOR: [0.1]}),
                SimpleNamespace(payload=new, vector={TEXT_VECTOR: [0.2]}),
            ],
            None,
        )


def test_stable_result_id_resolves_to_latest_published_generation() -> None:
    client = StableResultLookupClient()
    store = RagStore(Settings(), client=client)
    result_id = _versioned_doc("same", 1, ingestion_id="generation-old").id

    payload, vectors = store._payload_and_vectors_for_result_id(result_id)

    assert payload is not None
    assert payload["ingestion_id"] == "generation-new"
    assert vectors == {TEXT_VECTOR: [0.2]}
    conditions = {condition.key: condition for condition in client.scroll_filter.must}
    assert conditions["id"].match.value == result_id
    assert client.scroll_filter.must_not[0].key == "is_staged"


class FakeQdrantClient:
    def __init__(self) -> None:
        self.hits = [
            _hit("old", "111-2018.pdf", "111", 2018, 0.95),
            _hit("new", "222-2025.pdf", "222", 2025, 0.80),
            _hit("middle", "333-2023.pdf", "333", 2023, 0.90),
        ]

    def query_points(self, **_kwargs):
        return SimpleNamespace(points=self.hits)


def _hit(point_id: str, source_pdf: str, team: str, year: int, score: float):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "id": point_id,
            "source_id": source_pdf.removesuffix(".pdf"),
            "source_pdf": source_pdf,
            "team": team,
            "year": year,
            "page": 1,
            "modality": "text",
            "text": "shooter",
            "artifact_path": None,
            "linked_artifacts": [],
        },
    )


def test_search_can_sort_candidates_by_year() -> None:
    store = RagStore(Settings(), client=FakeQdrantClient())

    newest, newest_coverage = store.search(
        SearchRequest(query="shooter", top_k=2, sort="newest"),
        [0.0],
        [0.0],
        "shooter",
    )
    oldest, oldest_coverage = store.search(
        SearchRequest(query="shooter", top_k=2, sort="oldest"),
        [0.0],
        [0.0],
        "shooter",
    )

    assert [result.year for result in newest] == [2025, 2023]
    assert [result.year for result in oldest] == [2018, 2023]
    assert newest_coverage.returned_pages == 2
    assert oldest_coverage.weak_pages_dropped == 0


def test_search_drops_uniformly_weak_candidates() -> None:
    client = FakeQdrantClient()
    client.hits = [
        _hit("weak-one", "111-2018.pdf", "111", 2018, 0.10),
        _hit("weak-two", "222-2025.pdf", "222", 2025, 0.34),
    ]
    client.hits[1].payload["text"] = "swerve azimuth backlash"
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.35), client=client)

    results, coverage = store.search(
        SearchRequest(query="swerve azimuth backlash", top_k=10),
        [0.0],
        [0.0],
        "swerve azimuth backlash",
    )

    assert results == []
    assert coverage.candidate_pages == 2
    assert coverage.weak_pages_dropped == 2
    assert coverage.returned_pages == 0


class MixedScoreQdrantClient:
    def query_points(self, **kwargs):
        if kwargs["using"] == "text":
            hit = _hit("weak", "254-2023.pdf", "254", 2023, 0.34)
            hit.payload["text"] = (
                "compact fast reliable floor cone cube intake roller belt pivot compliant "
                "packaging geometry mechanism"
            )
        else:
            hit = _hit("eligible", "254-2023.pdf", "254", 2023, 0.40)
            hit.payload["text"] = "collector"
        return SimpleNamespace(points=[hit])


def test_eligible_hit_survives_weak_page_duplicate_with_lexical_bonus() -> None:
    store = RagStore(
        Settings(SEARCH_MIN_SCORE=0.35),
        client=MixedScoreQdrantClient(),
    )
    query = (
        "compact fast reliable floor cone cube intake roller belt pivot compliant packaging "
        "geometry mechanism"
    )

    results, coverage = store.search(
        SearchRequest(query=query),
        [0.0],
        [0.0],
        query,
    )

    assert [result.id for result in results] == ["eligible"]
    assert coverage.candidate_pages == 1
    assert coverage.weak_pages_dropped == 0


class SamePointMixedScoreClient:
    def query_points(self, **kwargs):
        if kwargs["using"] == "text":
            hit = _hit("same-point", "254-2023.pdf", "254", 2023, 0.34)
            hit.payload["text"] = "swerve azimuth backlash reduction module"
        else:
            hit = _hit("same-point", "254-2023.pdf", "254", 2023, 0.40)
            hit.payload["text"] = "swerve module"
        return SimpleNamespace(points=[hit])


def test_search_gates_a_merged_point_on_its_best_raw_vector_score() -> None:
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.35), client=SamePointMixedScoreClient())

    results, coverage = store.search(
        SearchRequest(query="swerve azimuth backlash reduction module"),
        [0.0],
        [0.0],
        "swerve azimuth backlash reduction module",
    )

    assert [result.id for result in results] == ["same-point"]
    assert coverage.weak_pages_dropped == 0


def test_find_similar_drops_hits_below_relevance_floor() -> None:
    client = FakeQdrantClient()
    client.hits = [_hit("weak", "111-2018.pdf", "111", 2018, 0.10)]
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.35), client=client)

    response = store._similar_from_vector(
        TEXT_VECTOR,
        [0.0],
        10,
        {"id": "seed", "source_pdf": "seed.pdf", "page": 1, "modality": "text"},
    )

    assert response.results == []
    assert response.coverage.candidate_pages == 1
    assert response.coverage.weak_pages_dropped == 1
    assert response.coverage.returned_pages == 0


class DualVectorSimilarityClient:
    def __init__(self) -> None:
        self.queried_vectors: list[str] = []

    def query_points(self, **kwargs):
        self.queried_vectors.append(kwargs["using"])
        scores = {
            TEXT_VECTOR: {
                "both": 0.70,
                "text-only": 0.60,
                "shape-only": 0.10,
            },
            IMAGE_VECTOR: {
                "both": 0.80,
                "text-only": 0.10,
                "shape-only": 0.75,
            },
        }
        return SimpleNamespace(
            points=[
                _hit(point_id, f"{point_id}.pdf", "254", 2023, score)
                for point_id, score in scores[kwargs["using"]].items()
            ]
        )


class ResultSimilarityClient(DualVectorSimilarityClient):
    def __init__(self) -> None:
        super().__init__()
        self.scroll_calls = 0

    def scroll(self, **kwargs):
        self.scroll_calls += 1
        conditions = {condition.key: condition for condition in kwargs["scroll_filter"].must}
        seed = _hit("seed", "seed.pdf", "254", 2023, 1.0)
        if "id" in conditions:
            seed.vector = {TEXT_VECTOR: [0.1], IMAGE_VECTOR: [0.0]}
            return ([seed], None)
        seed.payload["id"] = "seed-page-image"
        seed.payload["modality"] = "page_image"
        seed.vector = {TEXT_VECTOR: [0.1], IMAGE_VECTOR: [0.2]}
        return (
            [seed],
            None,
        )


def test_result_similarity_queries_both_seed_vectors() -> None:
    client = ResultSimilarityClient()
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.35), client=client)

    response = store.similar_from_result_id("seed", 10)

    assert response is not None
    assert client.scroll_calls == 2
    assert client.queried_vectors == [TEXT_VECTOR, IMAGE_VECTOR]
    assert [result.debug["similarity_reason"] for result in response.results] == [
        "both",
        "shape",
        "text",
    ]


class TextOnlyResultSimilarityClient(ResultSimilarityClient):
    def scroll(self, **kwargs):
        conditions = {condition.key: condition for condition in kwargs["scroll_filter"].must}
        if "id" not in conditions:
            self.scroll_calls += 1
            return [], None
        return super().scroll(**kwargs)


def test_result_similarity_skips_placeholder_image_vectors_without_a_page_image() -> None:
    client = TextOnlyResultSimilarityClient()
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.35), client=client)

    response = store.similar_from_result_id("seed", 10)

    assert response is not None
    assert client.queried_vectors == [TEXT_VECTOR]


class SnapshotBoundSimilarityStore(RagStore):
    def __init__(self) -> None:
        super().__init__(Settings(), client=SimpleNamespace())
        self._snapshots = iter(
            [
                {"254-2023": "generation-old"},
                {"254-2023": "generation-new"},
                {"254-2023": "generation-new"},
                {"254-2023": "generation-new"},
            ]
        )
        self.seed_snapshots: list[dict[str, str]] = []
        self.query_snapshots: list[dict[str, str]] = []

    def active_generations(self) -> dict[str, str]:
        return next(self._snapshots)

    def _similarity_seed_for_result_id(
        self,
        result_id: str,
        active_generations: dict[str, str],
    ) -> tuple[dict, dict]:
        self.seed_snapshots.append(active_generations)
        return (
            {
                "id": result_id,
                "source_id": "254-2023",
                "source_pdf": "254-2023.pdf",
                "page": 1,
                "modality": "text",
            },
            {TEXT_VECTOR: [0.1]},
        )

    def _similar_from_vectors(
        self,
        vectors: dict[str, list[float]],
        top_k: int,
        seed_payload: dict,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
        active_generations: dict[str, str] | None = None,
    ) -> SimilarPagesResponse:
        assert vectors == {TEXT_VECTOR: [0.1]}
        assert top_k == 10
        assert seed_payload["id"] == "seed"
        assert team_numbers is None
        assert years is None
        assert source_ids is None
        assert active_generations is not None
        self.query_snapshots.append(active_generations)
        return SimilarPagesResponse(seed=seed_payload, results=[])


def test_result_similarity_retries_seed_and_candidates_under_one_snapshot() -> None:
    store = SnapshotBoundSimilarityStore()

    response = store.similar_from_result_id("seed", 10)

    assert response is not None
    assert store.seed_snapshots == [
        {"254-2023": "generation-old"},
        {"254-2023": "generation-new"},
    ]
    assert store.query_snapshots == store.seed_snapshots


class SnapshotBoundPageSimilarityStore(RagStore):
    def __init__(self) -> None:
        super().__init__(Settings(), client=SimpleNamespace())
        self._snapshots = iter(
            [
                {"254-2023": "generation-old"},
                {"254-2023": "generation-new"},
                {"254-2023": "generation-new"},
                {"254-2023": "generation-new"},
            ]
        )
        self.current_snapshot: dict[str, str] = {}
        self.seed_snapshots: list[dict[str, str]] = []
        self.query_snapshots: list[dict[str, str]] = []

    def active_generations(self) -> dict[str, str]:
        self.current_snapshot = next(self._snapshots)
        return self.current_snapshot

    def _scroll_payloads(
        self,
        qfilter: models.Filter | None,
        limit: int | None,
        with_vectors: bool = False,
    ):
        assert qfilter is not None
        assert limit is None
        assert with_vectors is True
        snapshot = dict(self.current_snapshot)
        self.seed_snapshots.append(snapshot)
        return [
            (
                {
                    "id": "seed",
                    "source_id": "254-2023",
                    "source_pdf": "254-2023.pdf",
                    "ingestion_id": snapshot["254-2023"],
                    "page": 1,
                    "modality": "text",
                },
                {TEXT_VECTOR: [0.1]},
            )
        ]

    def _similar_from_vector(
        self,
        vector_name: str,
        vector: list[float],
        top_k: int,
        seed_payload: dict,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
        active_generations: dict[str, str] | None = None,
    ) -> SimilarPagesResponse:
        assert vector_name == TEXT_VECTOR
        assert vector == [0.1]
        assert top_k == 10
        assert seed_payload["id"] == "seed"
        assert team_numbers is None
        assert years is None
        assert source_ids is None
        assert active_generations is not None
        self.query_snapshots.append(active_generations)
        return SimilarPagesResponse(seed=seed_payload, results=[])


def test_page_similarity_retries_seed_and_candidates_under_one_snapshot() -> None:
    store = SnapshotBoundPageSimilarityStore()

    response = store.similar_from_page("254-2023.pdf", 1, 10)

    assert response is not None
    assert store.seed_snapshots == [
        {"254-2023": "generation-old"},
        {"254-2023": "generation-new"},
    ]
    assert store.query_snapshots == store.seed_snapshots


def test_find_similar_reports_text_shape_and_combined_matches() -> None:
    client = DualVectorSimilarityClient()
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.35), client=client)

    response = store._similar_from_vectors(
        {TEXT_VECTOR: [0.0], IMAGE_VECTOR: [0.0]},
        10,
        {"id": "seed", "source_pdf": "seed.pdf", "page": 1, "modality": "text"},
    )

    assert [result.id for result in response.results] == ["both", "shape-only", "text-only"]
    assert [result.debug["similarity_reason"] for result in response.results] == [
        "both",
        "shape",
        "text",
    ]
    assert response.coverage.candidate_pages == 3
    assert response.coverage.weak_pages_dropped == 0
    assert client.queried_vectors == [TEXT_VECTOR, IMAGE_VECTOR]


def test_find_similar_applies_exact_metadata_filters_before_retrieval() -> None:
    client = FakeQdrantClient()
    client.query_filter = None

    def query_points(**kwargs):
        client.query_filter = kwargs["query_filter"]
        return SimpleNamespace(points=[])

    client.query_points = query_points
    store = RagStore(Settings(), client=client)

    store._similar_from_vector(
        TEXT_VECTOR,
        [0.0],
        10,
        {"id": "seed", "source_pdf": "seed.pdf", "page": 1, "modality": "text"},
        team_numbers=["254"],
        years=[2023],
        source_ids=["254-2023"],
    )

    conditions = {condition.key: condition for condition in client.query_filter.must}
    assert conditions["team"].match.value == "254"
    assert conditions["year"].match.value == 2023
    assert conditions["source_id"].match.value == "254-2023"


def test_search_request_bounds_filter_cost() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="intake", team_numbers=[str(team) for team in range(21)])
    with pytest.raises(ValidationError):
        SearchRequest(query="intake", mechanism_types=["x" * 81])
    with pytest.raises(ValidationError):
        SearchRequest(query="intake", source_ids=["source"] * 21)


class PagingQdrantClient:
    def __init__(self) -> None:
        self.calls = 0
        self.payload_fields: list[str] = []
        self.scroll_filter = None

    def scroll(self, **kwargs):
        self.calls += 1
        self.payload_fields = kwargs["with_payload"]
        self.scroll_filter = kwargs["scroll_filter"]
        if kwargs.get("offset") is None:
            return (
                [
                    SimpleNamespace(
                        payload={
                            "source_id": "254-2023",
                            "source_pdf": "254-2023.pdf",
                            "team": "254",
                            "year": 2023,
                            "page": 1,
                            "modality": "text",
                        }
                    )
                ],
                "next-page",
            )
        return (
            [
                SimpleNamespace(
                    payload={
                        "source_id": "4414-2024",
                        "source_pdf": "4414-2024.pdf",
                        "team": "4414",
                        "year": 2024,
                        "page": 2,
                        "modality": "page_image",
                        "ingested_at": "2026-08-16T12:00:00+00:00",
                        "source_url": "https://example.com/4414-2024.pdf",
                    }
                )
            ],
            None,
        )


def test_list_sources_pushes_exact_filters_into_the_paginated_store_scan() -> None:
    client = PagingQdrantClient()
    store = RagStore(Settings(), client=client)

    response = store.list_sources(
        team_numbers=["4414"],
        years=[2024],
        source_ids=["4414-2024"],
        source_query="4414",
    )

    assert client.calls == 2
    assert "text" not in client.payload_fields
    assert "source_id" in client.payload_fields
    assert len(response.sources) == 1
    assert response.sources[0].source_id == "4414-2024"
    assert response.sources[0].page_image_count == 1
    assert response.sources[0].source_url == "https://example.com/4414-2024.pdf"
    conditions = {condition.key: condition for condition in client.scroll_filter.must}
    assert conditions["team"].match.value == "4414"
    assert conditions["year"].match.value == 2024
    assert conditions["source_id"].match.value == "4414-2024"
    assert client.scroll_filter.must_not[0].key == "is_staged"


def test_artifact_urls_encode_fragments_and_round_trip_through_static_files(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "generation#one" / "page.png"
    artifact.parent.mkdir()
    artifact.write_bytes(b"page-image")
    store = RagStore(Settings(ARTIFACT_DIR=tmp_path), client=SimpleNamespace())

    artifact_url = store._artifact_url(str(artifact))

    assert artifact_url == "/images/generation%23one/page.png"
    assert store._artifact_path_from_url(f"https://api.example.com{artifact_url}") == str(
        artifact.resolve()
    )
    static_app = FastAPI()
    static_app.mount("/images", StaticFiles(directory=tmp_path))
    response = TestClient(static_app).get(artifact_url)
    assert response.status_code == 200
    assert response.content == b"page-image"


def test_ingestion_control_state_is_outside_the_public_artifact_root(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    settings = Settings(ARTIFACT_DIR=artifact_dir)
    store = RagStore(settings, client=SimpleNamespace())

    store.set_active_generation("254-2023", "generation-a")

    assert settings.rag_state_dir.parent == artifact_dir.parent
    assert settings.rag_state_dir != artifact_dir
    assert (settings.rag_state_dir / "active-generations.json").is_file()
    static_app = FastAPI()
    static_app.mount("/images", StaticFiles(directory=artifact_dir, check_dir=False))
    response = TestClient(static_app).get("/images/active-generations.json")
    assert response.status_code == 404


def test_ingestion_control_state_rejects_a_public_subdirectory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    settings = Settings(
        ARTIFACT_DIR=artifact_dir,
        RAG_STATE_DIR=artifact_dir / "state",
    )

    with pytest.raises(ValueError, match="outside ARTIFACT_DIR"):
        _ = settings.rag_state_dir
