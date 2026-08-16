from app.rag.chunking import expand_query
from app.rag.config import Settings
from app.rag.models import SearchRequest, SearchResponse, SearchSort
from app.rag.store import RagStore
from app.rag.voyage_client import VoyageEmbedder


def search(
    query: str,
    top_k: int,
    debug: bool,
    settings: Settings,
    team: str | None = None,
    year: int | None = None,
    source: str | None = None,
    team_numbers: list[str] | None = None,
    years: list[int] | None = None,
    source_ids: list[str] | None = None,
    mechanism_types: list[str] | None = None,
    sort: SearchSort = "relevance",
    modality: str | None = None,
    store: RagStore | None = None,
) -> SearchResponse:
    request = SearchRequest(
        query=query,
        top_k=top_k,
        debug=debug,
        team=team,
        year=year,
        source=source,
        team_numbers=team_numbers or [],
        years=years or [],
        source_ids=source_ids or [],
        mechanism_types=mechanism_types or [],
        sort=sort,
        modality=modality,  # type: ignore[arg-type]
    )
    expanded = expand_query(
        " ".join([query, *request.mechanism_types]),
        request.years or ([request.year] if request.year else None),
    )
    embedder = VoyageEmbedder(settings)
    text_vector = embedder.embed_texts([expanded], "query")[0]
    image_vector = embedder.embed_multimodal([expanded], [None], "query")[0]
    results, coverage = (store or RagStore(settings)).search(
        request, text_vector, image_vector, expanded
    )
    abstention_reason = None
    if not results:
        abstention_reason = (
            "No indexed pages met the calibrated relevance threshold for this query and filter set."
            if coverage.candidate_pages
            else "No indexed pages matched this query and filter set."
        )
    return SearchResponse(
        query=query,
        results=results,
        coverage=coverage,
        abstention_reason=abstention_reason,
    )
