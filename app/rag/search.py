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
    return _execute_search(request, settings, store=store)


def search_source_catalog(
    query: str,
    top_k: int,
    settings: Settings,
    *,
    team_numbers: list[str],
    years: list[int],
    source_ids: list[str],
    store: RagStore | None = None,
) -> SearchResponse:
    """Search a trusted source catalog without applying the connector's 20-ID input bound."""
    request = SearchRequest(
        query=query,
        top_k=top_k,
        team_numbers=team_numbers,
        years=years,
    )
    return _execute_search(request, settings, store=store, source_ids=source_ids)


def _execute_search(
    request: SearchRequest,
    settings: Settings,
    *,
    store: RagStore | None,
    source_ids: list[str] | None = None,
) -> SearchResponse:
    expanded = expand_query(
        " ".join([request.query, *request.mechanism_types]),
        _expansion_years(request),
    )
    embedder = VoyageEmbedder(settings)
    text_vector = embedder.embed_texts([expanded], "query")[0]
    image_vector = embedder.embed_multimodal([expanded], [None], "query")[0]
    results, coverage = (store or RagStore(settings)).search(
        request,
        text_vector,
        image_vector,
        expanded,
        source_ids=source_ids,
    )
    abstention_reason = None
    if not results:
        abstention_reason = (
            "No indexed pages met the calibrated relevance threshold for this query and filter set."
            if coverage.candidate_pages
            else "No indexed pages matched this query and filter set."
        )
    return SearchResponse(
        query=request.query,
        results=results,
        coverage=coverage,
        abstention_reason=abstention_reason,
    )


def _expansion_years(request: SearchRequest) -> list[int] | None:
    years = list(
        dict.fromkeys([*([request.year] if request.year is not None else []), *request.years])
    )
    return years or None
