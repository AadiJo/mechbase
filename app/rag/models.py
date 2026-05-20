from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Modality = Literal["text", "page_image", "extracted_image"]


class SourceDoc(BaseModel):
    path: Path
    team: str | None
    year: int | None
    source_id: str


class RagDocument(BaseModel):
    id: str
    source_id: str
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    modality: Modality
    text: str = ""
    artifact_path: str | None = None
    linked_artifacts: list[str] = Field(default_factory=list)
    section: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    team: str | None = None
    year: int | None = None
    source: str | None = None
    modality: Modality | None = None
    debug: bool = False


class SearchResult(BaseModel):
    id: str
    score: float
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


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class PageContextResponse(BaseModel):
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
    source_pdf: str
    team: str | None = None
    year: int | None = None
    pages: list[int] = Field(default_factory=list)
    page_count: int = 0
    text_count: int = 0
    page_image_count: int = 0
    extracted_image_count: int = 0
    sample_image_urls: list[str] = Field(default_factory=list)


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
    source_pdf: str
    team: str | None = None
    year: int | None = None
    page: int
    page_context_url: str
    page_text_url: str
    text: str
    page_image_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
