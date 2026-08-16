from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.rag.config import Settings
from app.rag.models import RagDocument, SearchRequest
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


def test_build_filter_combines_legacy_and_multi_value_filters() -> None:
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
    team_conditions = [condition for condition in conditions if condition.key == "team"]
    year_conditions = [condition for condition in conditions if condition.key == "year"]
    source_condition = next(condition for condition in conditions if condition.key == "source_pdf")
    assert [condition.match.value for condition in team_conditions] == ["254", "4414"]
    assert [condition.match.value for condition in year_conditions] == [2023, 2024]
    assert source_condition.match.value == "254-2023.pdf"


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

    store.mark_corpus_revision("generation-new")

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


def test_staged_generation_is_hidden_from_source_and_page_reads() -> None:
    client = QdrantClient(":memory:")
    vector_params = models.VectorParams(size=1, distance=models.Distance.COSINE)
    client.create_collection(
        collection_name="frc_mechanisms",
        vectors_config={TEXT_VECTOR: vector_params, IMAGE_VECTOR: vector_params},
    )
    store = RagStore(Settings(EMBEDDING_DIM=1), client=client)
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

    summaries = store.list_sources().sources

    assert [summary.pages for summary in summaries] == [[1]]
    assert store.page_context("254-2023.pdf", 2) is None
    client.close()


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


def test_list_sources_paginates_and_filters_by_source_query() -> None:
    client = PagingQdrantClient()
    store = RagStore(Settings(), client=client)

    response = store.list_sources(source_query="4414")

    assert client.calls == 2
    assert "text" not in client.payload_fields
    assert "source_id" in client.payload_fields
    assert len(response.sources) == 1
    assert response.sources[0].source_id == "4414-2024"
    assert response.sources[0].page_image_count == 1
    assert response.sources[0].source_url == "https://example.com/4414-2024.pdf"
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
