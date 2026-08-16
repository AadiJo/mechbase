from types import SimpleNamespace

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

    newest = store.search(
        SearchRequest(query="shooter", top_k=2, sort="newest"),
        [0.0],
        [0.0],
        "shooter",
    )
    oldest = store.search(
        SearchRequest(query="shooter", top_k=2, sort="oldest"),
        [0.0],
        [0.0],
        "shooter",
    )

    assert [result.year for result in newest] == [2025, 2023]
    assert [result.year for result in oldest] == [2018, 2023]
