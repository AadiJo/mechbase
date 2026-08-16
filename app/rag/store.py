import json
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote, unquote, urlsplit
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.rag.config import Settings
from app.rag.models import (
    ImageContextResponse,
    PageContextResponse,
    RagDocument,
    ScoreBand,
    SearchCoverage,
    SearchRequest,
    SearchResult,
    SimilarPagesResponse,
    SourceListResponse,
    SourcePageMatch,
    SourceSummary,
)

TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"
MAX_RETRIEVAL_PAGES = 20
ACTIVE_GENERATIONS_FILE = "active-generations.json"
LEGACY_GENERATION = "__legacy__"
ReadValue = TypeVar("ReadValue")
SOURCE_SUMMARY_FIELDS = [
    "source_id",
    "source_version",
    "source_version_id",
    "ingestion_id",
    "source_pdf",
    "team",
    "year",
    "page",
    "modality",
    "artifact_path",
    "ingested_at",
    "source_url",
]
PAYLOAD_INDEXES = {
    "id": models.PayloadSchemaType.KEYWORD,
    "team": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "source_version": models.PayloadSchemaType.KEYWORD,
    "source_version_id": models.PayloadSchemaType.KEYWORD,
    "ingestion_id": models.PayloadSchemaType.KEYWORD,
    "is_staged": models.PayloadSchemaType.BOOL,
    "source_pdf": models.PayloadSchemaType.KEYWORD,
    "modality": models.PayloadSchemaType.KEYWORD,
    "artifact_path": models.PayloadSchemaType.KEYWORD,
    "linked_artifacts": models.PayloadSchemaType.KEYWORD,
}


@dataclass
class _SourceAccumulator:
    source_id: str
    source_version: str | None
    source_version_id: str
    ingestion_id: str | None
    source_pdf: str
    team: str | None
    year: int | None
    ingested_at: str | None
    source_url: str | None
    pages: set[int] = field(default_factory=set)
    text_count: int = 0
    page_image_count: int = 0
    extracted_image_count: int = 0
    sample_image_urls: list[str] = field(default_factory=list)


class RagStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
        )

    def ensure_collection(self) -> None:
        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.settings.collection_name not in existing:
            vector_params = models.VectorParams(
                size=self.settings.embedding_dim,
                distance=models.Distance.COSINE,
            )
            self.client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config={TEXT_VECTOR: vector_params, IMAGE_VECTOR: vector_params},
            )

        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> bool:
        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.settings.collection_name not in existing:
            return False

        collection = self.client.get_collection(self.settings.collection_name)
        indexed_fields = set((getattr(collection, "payload_schema", None) or {}).keys())
        for field_name, field_schema in PAYLOAD_INDEXES.items():
            if field_name in indexed_fields:
                continue
            self.client.create_payload_index(
                collection_name=self.settings.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        return True

    def corpus_revision(self) -> str:
        self.client.get_collection(self.settings.collection_name)
        manifest_path = self.settings.rag_state_dir / "ingestion-manifest.jsonl"
        active_path = self.settings.rag_state_dir / ACTIVE_GENERATIONS_FILE
        try:
            manifest = manifest_path.stat()
            manifest_revision = f"{manifest.st_mtime_ns}:{manifest.st_size}"
        except FileNotFoundError:
            manifest_revision = "missing"
        try:
            active_revision = sha256(active_path.read_bytes()).hexdigest()[:16]
        except FileNotFoundError:
            active_revision = "missing"
        return f"{manifest_revision}:{active_revision}"

    def active_generations(self) -> dict[str, str]:
        path = self.settings.rag_state_dir / ACTIVE_GENERATIONS_FILE
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(raw, dict) or not all(
            isinstance(source_id, str) and isinstance(ingestion_id, str)
            for source_id, ingestion_id in raw.items()
        ):
            raise ValueError(f"{path} must contain a string-to-string JSON object.")
        return raw

    def set_active_generation(self, source_id: str, ingestion_id: str) -> None:
        active = self.active_generations()
        active[source_id] = ingestion_id
        path = self.settings.rag_state_dir / ACTIVE_GENERATIONS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{ingestion_id}.tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(active, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def initialize_active_generation(self, source_id: str) -> None:
        active = self.active_generations()
        if source_id in active:
            return
        payloads = self._scroll_payloads(
            _active_filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ]
            ),
            limit=None,
        )
        if payloads:
            generation = _latest_generation(payloads)
            current_payload = next(
                payload for payload in payloads if _generation_key(payload) == generation
            )
            ingestion_id = current_payload.get("ingestion_id") or LEGACY_GENERATION
        else:
            ingestion_id = LEGACY_GENERATION
        self.set_active_generation(source_id, str(ingestion_id))

    def _read_with_active_snapshot(
        self,
        read: Callable[[dict[str, str]], ReadValue],
    ) -> tuple[ReadValue, dict[str, str]]:
        for _attempt in range(3):
            active = self.active_generations()
            value = read(active)
            if active == self.active_generations():
                return value, active
        raise RuntimeError("The active source generation changed repeatedly; retry the read.")

    def upsert(
        self,
        docs: list[RagDocument],
        text_vectors: list[list[float]],
        image_vectors: list[list[float]],
    ) -> None:
        points: list[models.PointStruct] = []
        for doc, text_vector, image_vector in zip(docs, text_vectors, image_vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, doc.storage_id or doc.id)),
                    vector={TEXT_VECTOR: text_vector, IMAGE_VECTOR: image_vector},
                    payload=doc.model_dump(),
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.settings.collection_name, points=points, wait=True
            )

    def delete_superseded_source_generations(
        self,
        source_id: str,
        ingestion_id: str,
    ) -> None:
        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="ingestion_id",
                            match=models.MatchValue(value=ingestion_id),
                        )
                    ],
                )
            ),
            wait=True,
        )

    def delete_source_generation(self, source_id: str, ingestion_id: str) -> None:
        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        ),
                        models.FieldCondition(
                            key="ingestion_id",
                            match=models.MatchValue(value=ingestion_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def publish_source_generation(self, source_id: str, ingestion_id: str) -> None:
        self.client.set_payload(
            collection_name=self.settings.collection_name,
            payload={"is_staged": False},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    ),
                    models.FieldCondition(
                        key="ingestion_id",
                        match=models.MatchValue(value=ingestion_id),
                    ),
                ]
            ),
            wait=True,
        )

    def retire_superseded_source_generations(
        self,
        source_id: str,
        ingestion_id: str,
    ) -> None:
        self.client.set_payload(
            collection_name=self.settings.collection_name,
            payload={"is_staged": True},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="ingestion_id",
                        match=models.MatchValue(value=ingestion_id),
                    )
                ],
            ),
            wait=True,
        )

    def search(
        self,
        request: SearchRequest,
        text_vector: list[float],
        image_vector: list[float],
        expanded_query: str,
    ) -> tuple[list[SearchResult], SearchCoverage]:
        limit = max(request.top_k * 12, 60)
        (text_hits, image_hits), active_generations = self._read_with_active_snapshot(
            lambda active: (
                self.client.query_points(
                    collection_name=self.settings.collection_name,
                    query=text_vector,
                    using=TEXT_VECTOR,
                    query_filter=_build_filter(request, active),
                    limit=limit,
                    with_payload=True,
                ).points,
                self.client.query_points(
                    collection_name=self.settings.collection_name,
                    query=image_vector,
                    using=IMAGE_VECTOR,
                    query_filter=_build_filter(request, active),
                    limit=limit,
                    with_payload=True,
                ).points,
            )
        )
        merged: dict[str, tuple[float, dict, dict]] = {}
        for source, hits in [("text", text_hits), ("image", image_hits)]:
            for hit in hits:
                point_id = str(hit.id)
                payload = dict(hit.payload or {})
                if not _is_current_generation(payload, active_generations):
                    continue
                lexical = _lexical_bonus(expanded_query, payload.get("text", ""))
                vector_score = float(hit.score)
                score = vector_score + lexical
                existing = merged.get(point_id)
                max_vector_score = max(
                    vector_score,
                    float(existing[2]["max_vector_score"]) if existing else vector_score,
                )
                if existing is None or score > existing[0]:
                    merged[point_id] = (
                        score,
                        payload,
                        {
                            "vector_source": source,
                            "vector_score": vector_score,
                            "max_vector_score": max_vector_score,
                            "lexical_bonus": lexical,
                        },
                    )
                else:
                    existing[2]["max_vector_score"] = max_vector_score
        ranked = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
        candidate_pages: set[tuple[str, int]] = set()
        candidate_sources: set[str] = set()
        relevant_pages: set[tuple[str, int]] = set()
        ranked_results: list[SearchResult] = []
        for _point_id, (score, payload, debug) in ranked:
            page_key = _page_key(payload)
            source_id = str(payload.get("source_id") or Path(payload.get("source_pdf", "")).stem)
            candidate_pages.add(page_key)
            candidate_sources.add(source_id)
            vector_score = float(debug["max_vector_score"])
            if vector_score < self.settings.search_min_score or page_key in relevant_pages:
                continue
            relevant_pages.add(page_key)
            ranked_results.append(
                self._search_result_from_payload(
                    payload,
                    score,
                    debug if request.debug else {},
                    band_score=vector_score,
                )
            )

        weak_pages_dropped = len(candidate_pages - relevant_pages)

        if request.sort == "newest":
            ranked_results.sort(
                key=lambda result: (
                    result.year is not None,
                    result.year or 0,
                    result.score,
                ),
                reverse=True,
            )
        elif request.sort == "oldest":
            ranked_results.sort(
                key=lambda result: (
                    result.year is None,
                    result.year or 0,
                    -result.score,
                )
            )
        results = ranked_results[: request.top_k]
        return results, SearchCoverage(
            candidate_pages=len(candidate_pages),
            candidate_sources=len(candidate_sources),
            weak_pages_dropped=weak_pages_dropped,
            returned_pages=len(results),
            candidate_window_truncated=(len(text_hits) == limit or len(image_hits) == limit),
        )

    def page_context(
        self,
        source_pdf: str,
        page: int,
        source_version_id: str | None = None,
        ingestion_id: str | None = None,
    ) -> PageContextResponse | None:
        contexts = self.page_contexts(
            source_pdf,
            [page],
            source_version_id,
            ingestion_id,
        )
        return contexts[0] if contexts else None

    def page_contexts(
        self,
        source_pdf: str,
        pages: list[int] | None,
        source_version_id: str | None = None,
        ingestion_id: str | None = None,
    ) -> list[PageContextResponse]:
        if pages == []:
            return []
        conditions = [
            models.FieldCondition(key="source_pdf", match=models.MatchValue(value=source_pdf)),
        ]
        if pages is not None:
            conditions.append(_match_values("page", list(dict.fromkeys(pages))))
        if source_version_id:
            conditions.append(
                models.FieldCondition(
                    key="source_version_id",
                    match=models.MatchValue(value=source_version_id),
                )
            )
        if ingestion_id:
            conditions.append(
                models.FieldCondition(
                    key="ingestion_id",
                    match=models.MatchValue(value=ingestion_id),
                )
            )
        payloads, active_generations = self._read_with_active_snapshot(
            lambda active: self._scroll_payloads(
                _active_filter(must=conditions, active_generations=active),
                limit=None,
            )
        )
        payloads = [
            payload for payload in payloads if _is_current_generation(payload, active_generations)
        ]
        if not payloads:
            return []
        if ingestion_id is None:
            selected_generation = _latest_generation(payloads)
            payloads = [
                payload for payload in payloads if _generation_key(payload) == selected_generation
            ]

        grouped: dict[int, list[dict]] = {}
        for payload in payloads:
            payload_page = int(payload.get("page", 0))
            if payload_page:
                grouped.setdefault(payload_page, []).append(payload)
        return [
            self._page_context_from_payloads(source_pdf, page, grouped[page])
            for page in sorted(grouped)
        ]

    def _page_context_from_payloads(
        self,
        source_pdf: str,
        page: int,
        payloads: list[dict],
    ) -> PageContextResponse:
        text_payloads = [p for p in payloads if p.get("modality") == "text" and p.get("text")]
        if text_payloads:
            text_chunks = [p["text"] for p in sorted(text_payloads, key=_chunk_sort_key)]
            text = "\n\n".join(dict.fromkeys(text_chunks))
        else:
            text_chunks = []
            text = max((p.get("text", "") for p in payloads), key=len, default="")

        image_urls = []
        page_image_url = None
        for payload in payloads:
            artifact_path = payload.get("artifact_path")
            url = self._artifact_url(artifact_path)
            if url and url not in image_urls:
                image_urls.append(url)
            if payload.get("modality") == "page_image":
                page_image_url = url
            for linked in payload.get("linked_artifacts") or []:
                linked_url = self._artifact_url(linked)
                if linked_url and linked_url not in image_urls:
                    image_urls.append(linked_url)

        first = payloads[0]
        return PageContextResponse(
            source_id=first.get("source_id"),
            source_version=first.get("source_version"),
            source_version_id=first.get("source_version_id"),
            ingestion_id=first.get("ingestion_id"),
            source_pdf=source_pdf,
            source_url=first.get("source_url"),
            team=first.get("team"),
            year=first.get("year"),
            page=page,
            section=next(
                (payload.get("section") for payload in payloads if payload.get("section")), None
            ),
            text=text,
            text_chunks=text_chunks,
            page_image_url=page_image_url,
            image_urls=image_urls,
            result_ids=[payload.get("id", "") for payload in payloads if payload.get("id")],
            primary_result_id=next(
                (
                    payload.get("id")
                    for payload in payloads
                    if payload.get("modality") == "page_image" and payload.get("id")
                ),
                next((payload.get("id") for payload in payloads if payload.get("id")), None),
            ),
        )

    def list_sources(
        self,
        team: str | None = None,
        year: int | None = None,
        source: str | None = None,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
        source_query: str | None = None,
    ) -> SourceListResponse:
        payloads, active_generations = self._read_with_active_snapshot(
            lambda active: list(
                self._iter_source_payloads(
                    _metadata_filter(
                        team=team,
                        year=year,
                        source=source,
                        team_numbers=team_numbers,
                        years=years,
                        source_ids=source_ids,
                        active_generations=active,
                    )
                )
            )
        )
        current_payloads = (
            payload for payload in payloads if _is_current_generation(payload, active_generations)
        )
        summaries = _latest_source_summaries(self._summarize_sources(current_payloads))
        if source_query:
            needle = source_query.casefold()
            summaries = [
                summary
                for summary in summaries
                if needle in summary.source_id.casefold()
                or needle in summary.source_pdf.casefold()
                or needle in summary.source_version_id.casefold()
            ]
        return SourceListResponse(sources=summaries)

    def source_summary(self, source_pdf: str) -> SourceSummary | None:
        payloads, active_generations = self._read_with_active_snapshot(
            lambda active: list(
                self._iter_source_payloads(
                    _metadata_filter(source=source_pdf, active_generations=active)
                )
            )
        )
        current_payloads = (
            payload for payload in payloads if _is_current_generation(payload, active_generations)
        )
        summaries = self._summarize_sources(current_payloads)
        return (
            max(
                summaries,
                key=lambda summary: (summary.ingested_at or "", summary.ingestion_id or ""),
            )
            if summaries
            else None
        )

    def source_search_from_results(
        self, query: str | None, results: list[SearchResult]
    ) -> list[SourcePageMatch]:
        grouped: dict[tuple[str, int], SourcePageMatch] = {}
        for result in results:
            key = (result.ingestion_id or result.source_version_id, result.page)
            existing = grouped.get(key)
            snippet = result.text[:500] if result.text else ""
            image_urls = [url for url in [result.artifact_url, *result.linked_artifact_urls] if url]
            if existing is None:
                grouped[key] = SourcePageMatch(
                    source_version_id=result.source_version_id,
                    ingestion_id=result.ingestion_id,
                    source_pdf=result.source_pdf,
                    team=result.team,
                    year=result.year,
                    page=result.page,
                    score=result.score,
                    best_snippets=[snippet] if snippet else [],
                    image_urls=list(dict.fromkeys(image_urls)),
                    page_context_url=result.page_context_url,
                    page_text_url=result.page_text_url,
                )
                continue
            existing.score = max(existing.score, result.score)
            if (
                snippet
                and snippet not in existing.best_snippets
                and len(existing.best_snippets) < 3
            ):
                existing.best_snippets.append(snippet)
            existing.image_urls = list(dict.fromkeys([*existing.image_urls, *image_urls]))
        return sorted(grouped.values(), key=lambda match: match.score, reverse=True)

    def similar_from_result_id(
        self,
        result_id: str,
        top_k: int,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
    ) -> SimilarPagesResponse | None:
        payload, vectors = self._similarity_seed_for_result_id(result_id)
        if payload is None or vectors is None:
            return None
        available_vectors = {
            name: vector
            for name in (TEXT_VECTOR, IMAGE_VECTOR)
            if (vector := vectors.get(name)) is not None and _vector_has_signal(vector)
        }
        if not available_vectors:
            return None
        return self._similar_from_vectors(
            available_vectors,
            top_k,
            payload,
            team_numbers=team_numbers,
            years=years,
            source_ids=source_ids,
        )

    def similar_from_page(
        self, source_pdf: str, page: int, top_k: int
    ) -> SimilarPagesResponse | None:
        conditions = [
            models.FieldCondition(key="source_pdf", match=models.MatchValue(value=source_pdf)),
            models.FieldCondition(key="page", match=models.MatchValue(value=page)),
            models.FieldCondition(key="modality", match=models.MatchValue(value="text")),
        ]
        payloads, active_generations = self._read_with_active_snapshot(
            lambda active: self._scroll_payloads(
                _active_filter(must=conditions, active_generations=active),
                limit=None,
                with_vectors=True,
            )
        )
        if not payloads:
            return None
        payloads = [
            (payload, vectors)
            for payload, vectors in payloads
            if _is_current_generation(payload, active_generations)
        ]
        if not payloads:
            return None
        payload, vectors = payloads[0]
        vector = vectors.get(TEXT_VECTOR) if vectors else None
        if vector is None:
            return None
        return self._similar_from_vector(TEXT_VECTOR, vector, top_k, payload)

    def _similarity_seed_for_result_id(
        self,
        result_id: str,
    ) -> tuple[dict | None, dict | None]:
        conditions = [
            models.FieldCondition(
                key="id",
                match=models.MatchValue(value=result_id),
            )
        ]

        def read(active: dict[str, str]) -> tuple[dict | None, dict | None]:
            matches = self._scroll_payloads(
                _active_filter(must=conditions, active_generations=active),
                limit=None,
                with_vectors=True,
            )
            matches = [
                (payload, vectors)
                for payload, vectors in matches
                if _is_current_generation(payload, active)
            ]
            if not matches:
                return None, None
            latest_generation = _latest_generation(payload for payload, _vectors in matches)
            payload, raw_vectors = next(
                (candidate, vectors)
                for candidate, vectors in matches
                if _generation_key(candidate) == latest_generation
            )
            vectors = dict(raw_vectors or {})

            if payload.get("modality") == "text":
                vectors.pop(IMAGE_VECTOR, None)
                page_conditions: list[models.Condition] = [
                    models.FieldCondition(
                        key="source_pdf",
                        match=models.MatchValue(value=str(payload.get("source_pdf") or "")),
                    ),
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=int(payload.get("page", 0))),
                    ),
                    models.FieldCondition(
                        key="modality",
                        match=models.MatchValue(value="page_image"),
                    ),
                ]
                for key in ("source_version_id", "ingestion_id"):
                    if value := payload.get(key):
                        page_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value),
                            )
                        )
                page_images = self._scroll_payloads(
                    _active_filter(must=page_conditions, active_generations=active),
                    limit=None,
                    with_vectors=True,
                )
                page_image_vector = next(
                    (
                        candidate_vectors.get(IMAGE_VECTOR)
                        for candidate, candidate_vectors in page_images
                        if _is_current_generation(candidate, active)
                        and candidate_vectors.get(IMAGE_VECTOR) is not None
                        and _vector_has_signal(candidate_vectors[IMAGE_VECTOR])
                    ),
                    None,
                )
                if page_image_vector is not None:
                    vectors[IMAGE_VECTOR] = page_image_vector
            return payload, vectors

        value, _active_generations = self._read_with_active_snapshot(read)
        return value

    def image_context(
        self,
        result_id: str | None = None,
        image_url: str | None = None,
    ) -> ImageContextResponse | None:
        payload = None
        if result_id:
            payload, _ = self._payload_and_vectors_for_result_id(result_id)
        elif image_url:
            artifact_path = self._artifact_path_from_url(image_url)
            if artifact_path:
                artifact_conditions = [
                    models.FieldCondition(
                        key="artifact_path", match=models.MatchValue(value=artifact_path)
                    ),
                    models.FieldCondition(
                        key="linked_artifacts", match=models.MatchAny(any=[artifact_path])
                    ),
                ]
                payloads, active_generations = self._read_with_active_snapshot(
                    lambda active: self._scroll_payloads(
                        _active_filter(
                            should=artifact_conditions,
                            active_generations=active,
                        ),
                        limit=None,
                    )
                )
                payload = next(
                    (
                        candidate
                        for candidate in payloads
                        if _is_current_generation(candidate, active_generations)
                    ),
                    None,
                )
        if payload is None:
            return None
        source_pdf = payload.get("source_pdf", "")
        page = int(payload.get("page", 0))
        source_version_id = payload.get("source_version_id")
        ingestion_id = payload.get("ingestion_id")
        context = self.page_context(source_pdf, page, source_version_id, ingestion_id)
        if context is None:
            return None
        return ImageContextResponse(
            result_id=payload.get("id"),
            image_url=self._artifact_url(payload.get("artifact_path")) or image_url,
            source_id=payload.get("source_id"),
            source_version=payload.get("source_version"),
            source_version_id=source_version_id,
            ingestion_id=ingestion_id,
            source_pdf=source_pdf,
            source_url=payload.get("source_url"),
            team=payload.get("team"),
            year=payload.get("year"),
            page=page,
            page_context_url=self._page_context_url(source_pdf, page),
            page_text_url=self._page_text_url(source_pdf, page),
            text=context.text,
            page_image_url=context.page_image_url,
            image_urls=context.image_urls,
        )

    def _similar_from_vector(
        self,
        vector_name: str,
        vector: list[float],
        top_k: int,
        seed_payload: dict,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
    ) -> SimilarPagesResponse:
        return self._similar_from_vectors(
            {vector_name: vector},
            top_k,
            seed_payload,
            team_numbers=team_numbers,
            years=years,
            source_ids=source_ids,
        )

    def _similar_from_vectors(
        self,
        vectors: dict[str, list[float]],
        top_k: int,
        seed_payload: dict,
        *,
        team_numbers: list[str] | None = None,
        years: list[int] | None = None,
        source_ids: list[str] | None = None,
    ) -> SimilarPagesResponse:
        top_k = min(max(top_k, 1), MAX_RETRIEVAL_PAGES)
        limit = max(top_k * 12, 60)
        hits_by_vector, active_generations = self._read_with_active_snapshot(
            lambda active: {
                vector_name: self.client.query_points(
                    collection_name=self.settings.collection_name,
                    query=vector,
                    using=vector_name,
                    query_filter=_metadata_filter(
                        team_numbers=team_numbers,
                        years=years,
                        source_ids=source_ids,
                        active_generations=active,
                    ),
                    limit=limit,
                    with_payload=True,
                ).points
                for vector_name, vector in vectors.items()
            }
        )
        page_hits: dict[tuple[str, int], dict[str, tuple[float, dict]]] = {}
        candidate_pages: set[tuple[str, int]] = set()
        candidate_sources: set[str] = set()
        seed_page = _page_key(seed_payload)
        for vector_name, hits in hits_by_vector.items():
            for hit in hits:
                payload = dict(hit.payload or {})
                if not _is_current_generation(payload, active_generations):
                    continue
                page_key = _page_key(payload)
                if page_key == seed_page:
                    continue
                candidate_pages.add(page_key)
                candidate_sources.add(
                    str(payload.get("source_id") or Path(payload.get("source_pdf", "")).stem)
                )
                vector_hits = page_hits.setdefault(page_key, {})
                score = float(hit.score)
                existing = vector_hits.get(vector_name)
                if existing is None or score > existing[0]:
                    vector_hits[vector_name] = (score, payload)

        ranked_pages: list[tuple[float, dict, str, dict[str, float]]] = []
        for vector_hits in page_hits.values():
            relevant_hits = {
                name: hit
                for name, hit in vector_hits.items()
                if hit[0] >= self.settings.search_min_score
            }
            if not relevant_hits:
                continue
            best_name, (best_score, best_payload) = max(
                relevant_hits.items(),
                key=lambda item: item[1][0],
            )
            if TEXT_VECTOR in relevant_hits and IMAGE_VECTOR in relevant_hits:
                reason = "both"
            elif best_name == IMAGE_VECTOR:
                reason = "shape"
            else:
                reason = "text"
            ranked_pages.append(
                (
                    best_score,
                    best_payload,
                    reason,
                    {name: hit[0] for name, hit in vector_hits.items()},
                )
            )

        ranked_pages.sort(key=lambda item: item[0], reverse=True)
        results = [
            self._search_result_from_payload(
                payload,
                score,
                {
                    "vector_source": reason,
                    "similarity_reason": reason,
                    "vector_scores": vector_scores,
                },
            )
            for score, payload, reason, vector_scores in ranked_pages[:top_k]
        ]
        relevant_pages = {
            _page_key(payload) for _score, payload, _reason, _scores in ranked_pages
        }
        return SimilarPagesResponse(
            seed={
                "id": seed_payload.get("id"),
                "source_pdf": seed_payload.get("source_pdf"),
                "page": seed_payload.get("page"),
                "modality": seed_payload.get("modality"),
            },
            results=results,
            coverage=SearchCoverage(
                candidate_pages=len(candidate_pages),
                candidate_sources=len(candidate_sources),
                weak_pages_dropped=len(candidate_pages - relevant_pages),
                returned_pages=len(results),
                candidate_window_truncated=any(
                    len(hits) == limit for hits in hits_by_vector.values()
                ),
            ),
        )

    def _search_result_from_payload(
        self,
        payload: dict,
        score: float,
        debug: dict | None = None,
        *,
        band_score: float | None = None,
    ) -> SearchResult:
        return SearchResult(
            id=str(payload.get("id") or ""),
            score=score,
            score_band=_score_band(
                score if band_score is None else band_score,
                self.settings.search_min_score,
            ),
            source_id=str(payload.get("source_id") or Path(payload.get("source_pdf", "")).stem),
            source_version=payload.get("source_version"),
            source_version_id=str(
                payload.get("source_version_id")
                or payload.get("source_id")
                or Path(payload.get("source_pdf", "")).stem
            ),
            ingestion_id=payload.get("ingestion_id"),
            source_pdf=payload.get("source_pdf", ""),
            team=payload.get("team"),
            year=payload.get("year"),
            page=int(payload.get("page", 0)),
            modality=payload.get("modality", ""),
            text=payload.get("text", ""),
            artifact_path=payload.get("artifact_path"),
            artifact_url=self._artifact_url(payload.get("artifact_path")),
            linked_artifacts=payload.get("linked_artifacts") or [],
            linked_artifact_urls=[
                url
                for path in payload.get("linked_artifacts") or []
                if (url := self._artifact_url(path))
            ],
            page_context_url=self._page_context_url(
                payload.get("source_pdf", ""),
                int(payload.get("page", 0)),
            ),
            page_text_url=self._page_text_url(
                payload.get("source_pdf", ""),
                int(payload.get("page", 0)),
            ),
            debug=debug or {},
        )

    def _payload_and_vectors_for_result_id(self, result_id: str) -> tuple[dict | None, dict | None]:
        conditions = [
            models.FieldCondition(
                key="id",
                match=models.MatchValue(value=result_id),
            )
        ]
        matches, active_generations = self._read_with_active_snapshot(
            lambda active: self._scroll_payloads(
                _active_filter(must=conditions, active_generations=active),
                limit=None,
                with_vectors=True,
            )
        )
        if not matches:
            return None, None
        matches = [
            (payload, vectors)
            for payload, vectors in matches
            if _is_current_generation(payload, active_generations)
        ]
        if not matches:
            return None, None
        latest_generation = _latest_generation(payload for payload, _vectors in matches)
        return next(
            (payload, vectors)
            for payload, vectors in matches
            if _generation_key(payload) == latest_generation
        )

    def _iter_source_payloads(
        self,
        qfilter: models.Filter | None,
    ) -> Iterator[dict]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.settings.collection_name,
                scroll_filter=qfilter,
                limit=1024,
                offset=offset,
                with_payload=SOURCE_SUMMARY_FIELDS,
                with_vectors=False,
            )
            for point in points:
                yield dict(point.payload or {})
            if offset is None:
                return

    def _scroll_payloads(
        self,
        qfilter: models.Filter | None,
        limit: int | None,
        with_vectors: bool = False,
    ):
        output = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.settings.collection_name,
                scroll_filter=qfilter,
                limit=min(limit - len(output), 256) if limit is not None else 256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for point in points:
                payload = dict(point.payload or {})
                if with_vectors:
                    output.append((payload, dict(point.vector or {})))
                else:
                    output.append(payload)
            if offset is None or (limit is not None and len(output) >= limit):
                return output

    def _summarize_sources(self, payloads: Iterable[dict]) -> list[SourceSummary]:
        grouped: dict[tuple[str, str], _SourceAccumulator] = {}
        for payload in payloads:
            source_id = str(payload.get("source_id") or Path(payload.get("source_pdf", "")).stem)
            source_version_id = str(payload.get("source_version_id") or source_id)
            ingestion_id = payload.get("ingestion_id")
            generation_key = (source_id, str(ingestion_id or source_version_id))
            accumulator = grouped.get(generation_key)
            if accumulator is None:
                accumulator = _SourceAccumulator(
                    source_id=source_id,
                    source_version=payload.get("source_version"),
                    source_version_id=source_version_id,
                    ingestion_id=ingestion_id,
                    source_pdf=payload.get("source_pdf", ""),
                    team=payload.get("team"),
                    year=payload.get("year"),
                    ingested_at=payload.get("ingested_at"),
                    source_url=payload.get("source_url"),
                )
                grouped[generation_key] = accumulator
            accumulator.source_version = accumulator.source_version or payload.get("source_version")
            accumulator.source_url = accumulator.source_url or payload.get("source_url")
            accumulator.team = accumulator.team or payload.get("team")
            accumulator.year = accumulator.year or payload.get("year")
            incoming_ingested_at = payload.get("ingested_at")
            if incoming_ingested_at and (
                accumulator.ingested_at is None or incoming_ingested_at > accumulator.ingested_at
            ):
                accumulator.ingested_at = incoming_ingested_at
            page = int(payload.get("page", 0))
            if page:
                accumulator.pages.add(page)
            modality = payload.get("modality")
            if modality == "text":
                accumulator.text_count += 1
            elif modality == "page_image":
                accumulator.page_image_count += 1
            elif modality == "extracted_image":
                accumulator.extracted_image_count += 1
            url = self._artifact_url(payload.get("artifact_path"))
            if (
                url
                and url not in accumulator.sample_image_urls
                and len(accumulator.sample_image_urls) < 5
            ):
                accumulator.sample_image_urls.append(url)

        summaries = []
        for accumulator in grouped.values():
            pages = sorted(accumulator.pages)
            summaries.append(
                SourceSummary(
                    source_id=accumulator.source_id,
                    source_version=accumulator.source_version,
                    source_version_id=accumulator.source_version_id,
                    ingestion_id=accumulator.ingestion_id,
                    source_pdf=accumulator.source_pdf,
                    team=accumulator.team,
                    year=accumulator.year,
                    pages=pages,
                    page_count=len(pages),
                    text_count=accumulator.text_count,
                    page_image_count=accumulator.page_image_count,
                    extracted_image_count=accumulator.extracted_image_count,
                    sample_image_urls=accumulator.sample_image_urls,
                    ingested_at=accumulator.ingested_at,
                    source_url=accumulator.source_url,
                )
            )
        return sorted(
            summaries, key=lambda item: (item.team or "", item.year or 0, item.source_pdf)
        )

    def _artifact_path_from_url(self, image_url: str) -> str | None:
        request_path = unquote(urlsplit(image_url).path)
        prefixes = [
            urlsplit(self.settings.artifact_url_base).path.rstrip("/"),
            "/artifacts",
        ]
        for prefix in prefixes:
            if request_path.startswith(prefix + "/"):
                rel = request_path.removeprefix(prefix + "/")
                artifact_root = self.settings.artifact_dir.resolve()
                candidate = (artifact_root / rel).resolve()
                try:
                    candidate.relative_to(artifact_root)
                except ValueError:
                    return None
                return str(candidate)
        return None

    def _artifact_url(self, artifact_path: str | None) -> str | None:
        if not artifact_path:
            return None
        try:
            rel = Path(artifact_path).resolve().relative_to(self.settings.artifact_dir.resolve())
        except ValueError:
            return None
        encoded_path = quote(rel.as_posix(), safe="/")
        return f"{self.settings.artifact_url_base.rstrip('/')}/{encoded_path}"

    def _page_context_url(
        self,
        source_pdf: str,
        page: int,
    ) -> str:
        return f"/pages/{quote(source_pdf, safe='')}/{page}"

    def _page_text_url(
        self,
        source_pdf: str,
        page: int,
    ) -> str:
        return f"/pages/{quote(source_pdf, safe='')}/{page}/text"


def _metadata_filter(
    team: str | None = None,
    year: int | None = None,
    source: str | None = None,
    team_numbers: list[str] | None = None,
    years: list[int] | None = None,
    source_ids: list[str] | None = None,
    active_generations: dict[str, str] | None = None,
) -> models.Filter:
    conditions = []
    normalized_teams = list(dict.fromkeys([*([team] if team else []), *(team_numbers or [])]))
    if normalized_teams:
        conditions.append(_match_values("team", normalized_teams))
    normalized_years = list(dict.fromkeys([*([year] if year else []), *(years or [])]))
    if normalized_years:
        conditions.append(_match_values("year", normalized_years))
    if source:
        conditions.append(
            models.FieldCondition(key="source_pdf", match=models.MatchValue(value=source))
        )
    if source_ids:
        conditions.append(_match_values("source_id", list(dict.fromkeys(source_ids))))
    return _active_filter(must=conditions, active_generations=active_generations)


def _build_filter(
    request: SearchRequest,
    active_generations: dict[str, str] | None = None,
) -> models.Filter:
    metadata_filter = _metadata_filter(
        team=request.team,
        year=request.year,
        source=request.source,
        team_numbers=request.team_numbers,
        years=request.years,
        source_ids=request.source_ids,
        active_generations=active_generations,
    )
    conditions = list(metadata_filter.must or [])
    if request.modality:
        conditions.append(
            models.FieldCondition(key="modality", match=models.MatchValue(value=request.modality))
        )
    return models.Filter(must=conditions, must_not=metadata_filter.must_not)


def _active_filter(
    *,
    must: list[models.Condition] | None = None,
    should: list[models.Condition] | None = None,
    active_generations: dict[str, str] | None = None,
) -> models.Filter:
    conditions = list(must or [])
    if active_generations:
        conditions.append(_active_generation_condition(active_generations))
    return models.Filter(
        must=conditions,
        should=should,
        must_not=[
            models.FieldCondition(
                key="is_staged",
                match=models.MatchValue(value=True),
            )
        ],
    )


def _active_generation_condition(active_generations: dict[str, str]) -> models.Filter:
    source_ids = list(active_generations)
    branches: list[models.Condition] = []
    ingestion_ids = [
        ingestion_id
        for ingestion_id in active_generations.values()
        if ingestion_id != LEGACY_GENERATION
    ]
    if ingestion_ids:
        branches.append(_match_values("ingestion_id", ingestion_ids))

    legacy_source_ids = [
        source_id
        for source_id, ingestion_id in active_generations.items()
        if ingestion_id == LEGACY_GENERATION
    ]
    if legacy_source_ids:
        branches.append(
            models.Filter(
                must=[
                    _match_values("source_id", legacy_source_ids),
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="ingestion_id")),
                ]
            )
        )

    # Sources not yet represented in the registry retain their pre-registry behavior.
    branches.append(
        models.FieldCondition(
            key="source_id",
            match=models.MatchExcept.model_validate({"except": source_ids}),
        )
    )
    return models.Filter(should=branches)


def _match_values(key: str, values: list[str] | list[int]) -> models.FieldCondition:
    if len(values) == 1:
        return models.FieldCondition(key=key, match=models.MatchValue(value=values[0]))
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def _lexical_bonus(query: str, text: str) -> float:
    if not text:
        return 0.0
    q_terms = {term for term in query.lower().split() if len(term) > 2}
    t = text.lower()
    matches = sum(1 for term in q_terms if term in t)
    return min(0.2, matches * 0.015)


def _vector_has_signal(vector: list[float]) -> bool:
    return any(value != 0.0 for value in vector)


def _score_band(score: float, minimum: float) -> ScoreBand:
    if score >= minimum + 0.25:
        return "strong"
    if score >= minimum + 0.1:
        return "good"
    return "relevant"


def _chunk_sort_key(payload: dict) -> tuple[int, str]:
    chunk_index = payload.get("chunk_index")
    if isinstance(chunk_index, int):
        return chunk_index, str(payload.get("id", ""))
    match = re.search(r"_text_(\d+)$", str(payload.get("id", "")))
    return (int(match.group(1)) if match else 0, str(payload.get("id", "")))


def _page_key(payload: dict) -> tuple[str, int]:
    return _generation_key(payload), int(payload.get("page", 0))


def _generation_key(payload: dict) -> str:
    return str(
        payload.get("ingestion_id")
        or payload.get("source_version_id")
        or payload.get("source_pdf", "")
    )


def _is_current_generation(payload: dict, active_generations: dict[str, str]) -> bool:
    source_id = str(payload.get("source_id") or Path(payload.get("source_pdf", "")).stem)
    active_ingestion = active_generations.get(source_id)
    if active_ingestion is None:
        return True
    if active_ingestion == LEGACY_GENERATION:
        return payload.get("ingestion_id") is None
    return payload.get("ingestion_id") == active_ingestion


def _latest_generation(payloads: Iterable[dict]) -> str:
    ingested_at_by_generation: dict[str, str] = {}
    for payload in payloads:
        generation = _generation_key(payload)
        ingested_at_by_generation[generation] = max(
            ingested_at_by_generation.get(generation, ""),
            str(payload.get("ingested_at") or ""),
        )
    return max(
        ingested_at_by_generation,
        key=lambda generation: (ingested_at_by_generation[generation], generation),
    )


def _latest_source_summaries(summaries: Iterable[SourceSummary]) -> list[SourceSummary]:
    latest: dict[str, SourceSummary] = {}
    for summary in summaries:
        current = latest.get(summary.source_id)
        if current is None or (
            summary.ingested_at or "",
            summary.ingestion_id or summary.source_version_id,
        ) > (
            current.ingested_at or "",
            current.ingestion_id or current.source_version_id,
        ):
            latest[summary.source_id] = summary
    return list(latest.values())
