from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Modality = Literal["text", "page_image", "extracted_image"]
SearchSort = Literal["relevance", "newest", "oldest"]
ScoreBand = Literal["relevant", "good", "strong"]
TeamFilter = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\d{1,5}$")]
SeasonFilter = Annotated[int, Field(ge=1992, le=2100)]
SourceIdFilter = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
MechanismTypeFilter = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
LegacyFilter = Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]


class SourceDoc(BaseModel):
    path: Path
    team: str | None
    year: int | None
    source_id: str
    source_version: str
    source_version_id: str
    source_url: str | None = None


class RagDocument(BaseModel):
    id: str
    source_id: str
    source_version: str
    source_version_id: str
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    modality: Modality
    text: str = ""
    artifact_path: str | None = None
    linked_artifacts: list[str] = Field(default_factory=list)
    section: str | None = None
    source_url: str | None = None
    ingested_at: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(max_length=8000)
    top_k: int = Field(default=10, ge=1, le=100)
    team: LegacyFilter | None = None
    year: int | None = None
    source: LegacyFilter | None = None
    team_numbers: Annotated[list[TeamFilter], Field(max_length=20)] = Field(default_factory=list)
    years: Annotated[list[SeasonFilter], Field(max_length=20)] = Field(default_factory=list)
    source_ids: Annotated[list[SourceIdFilter], Field(max_length=20)] = Field(default_factory=list)
    mechanism_types: Annotated[list[MechanismTypeFilter], Field(max_length=8)] = Field(
        default_factory=list
    )
    sort: SearchSort = "relevance"
    modality: Modality | None = None
    debug: bool = False


class SearchResult(BaseModel):
    id: str
    score: float
    score_band: ScoreBand
    source_id: str
    source_version_id: str
    source_version: str | None = None
    source_pdf: str
    team: str | None
    year: int | None
    page: int
    modality: str
    text: str
    artifact_path: str | None
    artifact_url: str | None = None
    linked_artifacts: list[str]
    linked_artifact_urls: list[str] = Field(default_factory=list)
    page_context_url: str
    page_text_url: str
    debug: dict = Field(default_factory=dict)


class SearchCoverage(BaseModel):
    candidate_pages: int = 0
    candidate_sources: int = 0
    weak_pages_dropped: int = 0
    returned_pages: int = 0


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    coverage: SearchCoverage = Field(default_factory=SearchCoverage)
    abstention_reason: str | None = None


class PageContextResponse(BaseModel):
    source_id: str | None = None
    source_version: str | None = None
    source_version_id: str | None = None
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    text: str
    text_chunks: list[str] = Field(default_factory=list)
    page_image_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    result_ids: list[str] = Field(default_factory=list)


class PageTextResponse(BaseModel):
    source_pdf: str
    page: int
    text: str


class SourceSummary(BaseModel):
    source_id: str
    source_version_id: str
    source_version: str | None = None
    source_pdf: str
    team: str | None = None
    year: int | None = None
    pages: list[int] = Field(default_factory=list)
    page_count: int = 0
    text_count: int = 0
    page_image_count: int = 0
    extracted_image_count: int = 0
    sample_image_urls: list[str] = Field(default_factory=list)
    ingested_at: str | None = None
    source_url: str | None = None


class SourceListResponse(BaseModel):
    sources: list[SourceSummary]


class SourceSearchRequest(BaseModel):
    query: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    team: str | None = None
    year: int | None = None
    source: str | None = None


class SourcePageMatch(BaseModel):
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    score: float
    best_snippets: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    page_context_url: str
    page_text_url: str


class SourceSearchResponse(BaseModel):
    query: str | None = None
    matches: list[SourcePageMatch]


class SimilarPagesResponse(BaseModel):
    seed: dict
    results: list[SearchResult]


class ImageContextResponse(BaseModel):
    result_id: str | None = None
    image_url: str | None = None
    source_id: str | None = None
    source_version: str | None = None
    source_version_id: str | None = None
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    page_context_url: str
    page_text_url: str
    text: str
    page_image_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
