from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from app.rag.models import ImageContextResponse, SearchResult, SourceSummary


class SearchItem(BaseModel):
    id: str
    title: str
    url: str


class SearchOutput(BaseModel):
    results: list[SearchItem]


class FetchOutput(BaseModel):
    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, JsonValue] | None = None


class SourceItem(BaseModel):
    source_pdf: str
    team: str | None = None
    year: int | None = None
    pages: list[int] = Field(default_factory=list)
    page_count: int
    sample_image_urls: list[str] = Field(default_factory=list)


class SourceOutput(BaseModel):
    sources: list[SourceItem]


def search_output(results: list[SearchResult], public_base_url: str) -> SearchOutput:
    return SearchOutput(
        results=[
            SearchItem(
                id=result.id,
                title=_result_title(result.source_pdf, result.page, result.team),
                url=_result_url(result, public_base_url),
            )
            for result in results
            if result.id
        ]
    )


def fetch_output(
    context: ImageContextResponse, result_id: str, public_base_url: str
) -> FetchOutput:
    image_urls = [_absolute_url(public_base_url, url) for url in context.image_urls]
    canonical_url = _absolute_url(
        public_base_url,
        context.page_image_url or context.image_url or context.page_context_url,
    )
    return FetchOutput(
        id=result_id,
        title=_result_title(context.source_pdf, context.page, context.team),
        text=context.text,
        url=canonical_url,
        metadata={
            "source_pdf": context.source_pdf,
            "team": context.team,
            "year": context.year,
            "page": context.page,
            "image_urls": image_urls,
        },
    )


def source_output(
    sources: list[SourceSummary],
    public_base_url: str,
) -> SourceOutput:
    return SourceOutput(
        sources=[
            SourceItem(
                source_pdf=source.source_pdf,
                team=source.team,
                year=source.year,
                pages=source.pages,
                page_count=source.page_count,
                sample_image_urls=[
                    _absolute_url(public_base_url, url) for url in source.sample_image_urls
                ],
            )
            for source in sources
        ]
    )


def _result_title(source_pdf: str, page: int, team: str | None) -> str:
    team_prefix = f"Team {team}: " if team else ""
    return f"{team_prefix}{source_pdf}, page {page}"


def _result_url(result: SearchResult, public_base_url: str) -> str:
    relative_url = result.artifact_url
    if relative_url is None and result.linked_artifact_urls:
        relative_url = result.linked_artifact_urls[0]
    return _absolute_url(public_base_url, relative_url or result.page_context_url)


def _absolute_url(public_base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith(("https://", "http://")):
        return path_or_url
    return f"{public_base_url.rstrip('/')}/{path_or_url.lstrip('/')}"
