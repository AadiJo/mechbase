from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from app.mcp.images import preview_image_url
from app.rag.models import (
    ImageContextResponse,
    ScoreBand,
    SearchCoverage,
    SearchResponse,
    SearchResult,
    SearchSort,
    SourceSummary,
)


class EvidenceClassification(BaseModel):
    direct_source_text: bool
    visible_image: bool
    model_inference: bool = False
    missing: list[str] = Field(default_factory=list)


class SearchItem(BaseModel):
    id: str
    title: str
    url: str
    source_id: str
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


class SourceItem(BaseModel):
    source_id: str
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
    return SearchOutput(
        results=[
            SearchItem(
                id=result.id,
                title=_result_title(result.source_pdf, result.page, result.team),
                url=_result_url(result, public_base_url),
                source_id=result.source_id,
                source_pdf=result.source_pdf,
                team=result.team,
                year=result.year,
                page=result.page,
                snippet=_truncate(result.text.strip(), 500),
                score_band=result.score_band,
                evidence=EvidenceClassification(
                    direct_source_text=bool(result.text.strip()),
                    visible_image=result.modality in {"page_image", "extracted_image"}
                    and bool(result.artifact_url),
                    missing=[
                        label
                        for missing, label in (
                            (not result.text.strip(), "direct source text"),
                            (
                                result.modality not in {"page_image", "extracted_image"}
                                or not result.artifact_url,
                                "visible image evidence",
                            ),
                        )
                        if missing
                    ],
                ),
            )
            for result in response.results
            if result.id
        ],
        applied_filters=applied_filters or AppliedSearchFilters(),
        coverage=response.coverage,
        abstention_reason=response.abstention_reason,
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
                source_id=source.source_id,
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
        coverage_found=bool(sources),
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
