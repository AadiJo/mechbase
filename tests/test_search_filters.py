from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.rag.config import Settings
from app.rag.models import SearchRequest
from app.rag.store import RagStore, _build_filter


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
    conditions = {condition.key: condition for condition in qfilter.must or []}
    assert conditions["team"].match.any == ["254", "4414"]
    assert conditions["year"].match.any == [2023, 2024]
    assert conditions["source_pdf"].match.value == "254-2023.pdf"


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
    store = RagStore(Settings())
    store.client = FakeQdrantClient()

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
    store = RagStore(Settings(SEARCH_MIN_SCORE=0.25))
    client = FakeQdrantClient()
    client.hits = [
        _hit("weak-one", "111-2018.pdf", "111", 2018, 0.10),
        _hit("weak-two", "222-2025.pdf", "222", 2025, 0.15),
    ]
    store.client = client

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

    def scroll(self, **kwargs):
        self.calls += 1
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
    store = RagStore(Settings())
    client = PagingQdrantClient()
    store.client = client

    response = store.list_sources(source_query="4414")

    assert client.calls == 2
    assert len(response.sources) == 1
    assert response.sources[0].source_id == "4414-2024"
    assert response.sources[0].page_image_count == 1
    assert response.sources[0].source_url == "https://example.com/4414-2024.pdf"
