from __future__ import annotations

from urllib.parse import quote, urldefrag

from pydantic import BaseModel, Field, JsonValue

from app.mcp.images import preview_image_url
from app.rag.models import (
    ImageContextResponse,
    PageContextResponse,
    ScoreBand,
    SearchCoverage,
    SearchResponse,
    SearchResult,
    SearchSort,
    SimilarPagesResponse,
    SourceSummary,
)


class EvidenceClassification(BaseModel):
    direct_source_text: bool
    visible_image: bool
    missing: list[str] = Field(default_factory=list)


class SearchItem(BaseModel):
    id: str
    title: str
    url: str
    source_id: str
    source_version_id: str
    source_version: str | None = None
    ingestion_id: str | None = None
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    snippet: str = ""
    score_band: ScoreBand
    evidence: EvidenceClassification


class AppliedSearchFilters(BaseModel):
    team_numbers: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    mechanism_types: list[str] = Field(default_factory=list)
    sort: SearchSort = "relevance"


class SearchOutput(BaseModel):
    results: list[SearchItem]
    applied_filters: AppliedSearchFilters = Field(default_factory=AppliedSearchFilters)
    coverage: SearchCoverage = Field(default_factory=SearchCoverage)
    abstention_reason: str | None = None
    evidence_limits: list[str] = Field(
        default_factory=lambda: [
            "Binder evidence does not independently verify competition performance.",
            "Vector relevance does not establish comparative design quality.",
        ]
    )


class FetchOutput(BaseModel):
    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, JsonValue] | None = None
    pages: list[FetchedPage] = Field(default_factory=list)


class FetchedPage(BaseModel):
    page: int
    text: str
    url: str
    page_image_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    evidence: EvidenceClassification


class SimilarItem(SearchItem):
    similarity_reason: str


class SimilarOutput(BaseModel):
    seed: dict[str, JsonValue]
    results: list[SimilarItem]
    applied_filters: AppliedSearchFilters = Field(default_factory=AppliedSearchFilters)
    coverage: SearchCoverage = Field(default_factory=SearchCoverage)
    abstention_reason: str | None = None
    evidence_limits: list[str] = Field(
        default_factory=lambda: [
            "Similarity means related design evidence, not quality or performance.",
        ]
    )


class BrowsePage(BaseModel):
    result_id: str
    page: int
    section: str | None = None
    snippet: str = ""
    url: str
    image_url: str | None = None


class BrowseSourceOutput(BaseModel):
    source_id: str
    source_version_id: str
    source_pdf: str
    team: str | None = None
    year: int | None = None
    pages: list[BrowsePage]
    missing_pages: list[int] = Field(default_factory=list)
    scanned_pages: int = 0
    next_cursor_page: int | None = None
    truncated: bool = False


class SourceItem(BaseModel):
    source_id: str
    source_version_id: str
    source_version: str | None = None
    ingestion_id: str | None = None
    source_pdf: str
    team: str | None = None
    year: int | None = None
    pages: list[int] = Field(default_factory=list)
    page_count: int
    text_count: int
    page_image_count: int
    extracted_image_count: int
    sample_image_urls: list[str] = Field(default_factory=list)
    ingested_at: str | None = None
    source_url: str | None = None


class SourceOutput(BaseModel):
    sources: list[SourceItem]
    coverage_found: bool
    total_matching_sources: int
    truncated: bool


class VisualCandidate(BaseModel):
    id: str
    title: str
    text: str
    url: str
    image_url: str | None = None
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    has_image: bool


class InspectOutput(BaseModel):
    candidates: list[VisualCandidate]
    missing_ids: list[str] = Field(default_factory=list)


class RenderItem(BaseModel):
    id: str
    title: str
    url: str
    image_url: str
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int


class RenderOutput(BaseModel):
    results: list[RenderItem]


def search_output(
    response: SearchResponse,
    public_base_url: str,
    *,
    applied_filters: AppliedSearchFilters | None = None,
) -> SearchOutput:
    items = [
        SearchItem(
            id=result.id,
            title=_result_title(result.source_pdf, result.page, result.team),
            url=_result_url(result, public_base_url),
            source_id=result.source_id,
            source_version=result.source_version,
            source_version_id=result.source_version_id,
            ingestion_id=result.ingestion_id,
            source_pdf=result.source_pdf,
            team=result.team,
            year=result.year,
            page=result.page,
            snippet=_truncate(result.text.strip(), 500),
            score_band=result.score_band,
            evidence=EvidenceClassification(
                direct_source_text=bool(result.text.strip()),
                visible_image=bool(result.artifact_url or result.linked_artifact_urls),
                missing=[
                    label
                    for missing, label in (
                        (not result.text.strip(), "direct source text"),
                        (
                            not (result.artifact_url or result.linked_artifact_urls),
                            "visible image evidence",
                        ),
                    )
                    if missing
                ],
            ),
        )
        for result in response.results
        if result.id
    ]
    return SearchOutput(
        results=items,
        applied_filters=applied_filters or AppliedSearchFilters(),
        coverage=response.coverage.model_copy(update={"returned_pages": len(items)}),
        abstention_reason=response.abstention_reason,
    )


def fetch_output(
    context: ImageContextResponse,
    result_id: str,
    public_base_url: str,
    adjacent_contexts: list[PageContextResponse] | None = None,
) -> FetchOutput:
    image_urls = [_absolute_url(public_base_url, url) for url in context.image_urls]
    canonical_url = _page_citation_url(context, public_base_url)
    all_pages: list[ImageContextResponse | PageContextResponse] = [
        *(adjacent_contexts or []),
        context,
    ]
    all_pages.sort(key=lambda item: item.page)
    return FetchOutput(
        id=result_id,
        title=_result_title(context.source_pdf, context.page, context.team),
        text=context.text,
        url=canonical_url,
        metadata={
            "source_id": context.source_id,
            "source_version": context.source_version,
            "source_version_id": context.source_version_id,
            "ingestion_id": context.ingestion_id,
            "source_pdf": context.source_pdf,
            "team": context.team,
            "year": context.year,
            "page": context.page,
            "image_urls": image_urls,
        },
        pages=[_fetched_page(page, public_base_url) for page in all_pages],
    )


def similar_output(
    response: SimilarPagesResponse,
    public_base_url: str,
    *,
    applied_filters: AppliedSearchFilters,
) -> SimilarOutput:
    base = search_output(
        SearchResponse(
            query="",
            results=response.results,
            coverage=response.coverage,
            abstention_reason=(
                None
                if response.results
                else "No similar pages met the calibrated relevance threshold."
            ),
        ),
        public_base_url,
        applied_filters=applied_filters,
    )
    return SimilarOutput(
        seed=response.seed,
        results=[
            SimilarItem(
                **item.model_dump(),
                similarity_reason=str(result.debug.get("similarity_reason") or "text"),
            )
            for item, result in zip(base.results, response.results, strict=True)
        ],
        applied_filters=base.applied_filters,
        coverage=base.coverage,
        abstention_reason=base.abstention_reason,
    )


def browse_source_output(
    source: SourceSummary,
    contexts: list[PageContextResponse],
    public_base_url: str,
    *,
    include_previews: bool,
    missing_pages: list[int],
    scanned_pages: int,
    next_cursor_page: int | None,
    truncated: bool,
) -> BrowseSourceOutput:
    pages = []
    for context in contexts:
        if not context.primary_result_id:
            continue
        image_url = (
            _absolute_url(public_base_url, context.page_image_url)
            if include_previews and context.page_image_url
            else None
        )
        pages.append(
            BrowsePage(
                result_id=context.primary_result_id,
                page=context.page,
                section=context.section,
                snippet=_truncate(context.text.strip(), 500),
                url=_absolute_url(
                    public_base_url,
                    context.page_image_url
                    or next(iter(context.image_urls), None)
                    or _source_page_url(source.source_url, context.page)
                    or f"/pages/{quote(source.source_pdf, safe='')}/{context.page}",
                ),
                image_url=image_url,
            )
        )
    return BrowseSourceOutput(
        source_id=source.source_id,
        source_version_id=source.source_version_id,
        source_pdf=source.source_pdf,
        team=source.team,
        year=source.year,
        pages=pages,
        missing_pages=missing_pages,
        scanned_pages=scanned_pages,
        next_cursor_page=next_cursor_page,
        truncated=truncated,
    )


def _fetched_page(
    context: ImageContextResponse | PageContextResponse,
    public_base_url: str,
) -> FetchedPage:
    image_urls = [_absolute_url(public_base_url, url) for url in context.image_urls]
    page_image_url = (
        _absolute_url(public_base_url, context.page_image_url) if context.page_image_url else None
    )
    page_url = _page_citation_url(context, public_base_url)
    has_visible_image = bool(image_urls or page_image_url)
    return FetchedPage(
        page=context.page,
        text=context.text,
        url=page_url,
        page_image_url=page_image_url,
        image_urls=image_urls,
        evidence=EvidenceClassification(
            direct_source_text=bool(context.text.strip()),
            visible_image=has_visible_image,
            missing=[
                label
                for missing, label in (
                    (not context.text.strip(), "direct source text"),
                    (not has_visible_image, "visible image evidence"),
                )
                if missing
            ],
        ),
    )


def _page_citation_url(
    context: ImageContextResponse | PageContextResponse,
    public_base_url: str,
) -> str:
    citation = (
        context.page_image_url
        or next(iter(context.image_urls), None)
        or _source_page_url(context.source_url, context.page)
        or getattr(
            context,
            "page_context_url",
            f"/pages/{quote(context.source_pdf, safe='')}/{context.page}",
        )
    )
    return _absolute_url(public_base_url, citation)


def _source_page_url(source_url: str | None, page: int) -> str | None:
    if source_url is None:
        return None
    base_url, _fragment = urldefrag(source_url)
    return f"{base_url}#page={page}"


def source_output(
    sources: list[SourceSummary],
    public_base_url: str,
    *,
    total_matching_sources: int | None = None,
) -> SourceOutput:
    total = len(sources) if total_matching_sources is None else total_matching_sources
    return SourceOutput(
        sources=[
            SourceItem(
                source_id=source.source_id,
                source_version=source.source_version,
                source_version_id=source.source_version_id,
                ingestion_id=source.ingestion_id,
                source_pdf=source.source_pdf,
                team=source.team,
                year=source.year,
                pages=source.pages,
                page_count=source.page_count,
                text_count=source.text_count,
                page_image_count=source.page_image_count,
                extracted_image_count=source.extracted_image_count,
                sample_image_urls=[
                    _absolute_url(public_base_url, url) for url in source.sample_image_urls
                ],
                ingested_at=source.ingested_at,
                source_url=source.source_url,
            )
            for source in sources
        ],
        coverage_found=total > 0,
        total_matching_sources=total,
        truncated=len(sources) < total,
    )


def visual_candidate(
    context: ImageContextResponse,
    result_id: str,
    public_base_url: str,
    *,
    has_image: bool,
) -> VisualCandidate:
    relative_image_url = preview_image_url(context)
    image_url = _absolute_url(public_base_url, relative_image_url) if relative_image_url else None
    canonical_url = image_url or _absolute_url(public_base_url, context.page_context_url)
    return VisualCandidate(
        id=result_id,
        title=_result_title(context.source_pdf, context.page, context.team),
        text=_truncate(context.text, 6000),
        url=canonical_url,
        image_url=image_url,
        source_pdf=context.source_pdf,
        team=context.team,
        year=context.year,
        page=context.page,
        has_image=has_image,
    )


def render_output(
    contexts: list[tuple[str, ImageContextResponse]],
    public_base_url: str,
) -> RenderOutput:
    results = []
    for result_id, context in contexts:
        relative_image_url = preview_image_url(context)
        if relative_image_url is None:
            continue
        image_url = _absolute_url(public_base_url, relative_image_url)
        results.append(
            RenderItem(
                id=result_id,
                title=_result_title(context.source_pdf, context.page, context.team),
                url=image_url,
                image_url=image_url,
                source_pdf=context.source_pdf,
                team=context.team,
                year=context.year,
                page=context.page,
            )
        )
    return RenderOutput(results=results)


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


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n\n[Page text truncated]"
