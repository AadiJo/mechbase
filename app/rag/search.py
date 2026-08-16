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
    expanded = expand_query(" ".join([query, *request.mechanism_types]))
    embedder = VoyageEmbedder(settings)
    text_vector = embedder.embed_texts([expanded], "query")[0]
    image_vector = embedder.embed_multimodal([expanded], [None], "query")[0]
    results = RagStore(settings).search(request, text_vector, image_vector, expanded)
    return SearchResponse(query=query, results=results)
