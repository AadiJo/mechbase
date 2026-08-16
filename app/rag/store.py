import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
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
SOURCE_SUMMARY_FIELDS = [
    "source_id",
    "source_version",
    "source_version_id",
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
    "team": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "source_version": models.PayloadSchemaType.KEYWORD,
    "source_version_id": models.PayloadSchemaType.KEYWORD,
    "source_pdf": models.PayloadSchemaType.KEYWORD,
    "modality": models.PayloadSchemaType.KEYWORD,
}


@dataclass
class _SourceAccumulator:
    source_id: str
    source_version: str | None
    source_version_id: str
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
        self.client = client or QdrantClient(url=settings.qdrant_url)

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

    def corpus_revision(self) -> str:
        collection = self.client.get_collection(self.settings.collection_name)
        manifest_path = self.settings.artifact_dir / "ingestion-manifest.jsonl"
        try:
            manifest = manifest_path.stat()
            manifest_revision = f"{manifest.st_mtime_ns}:{manifest.st_size}"
        except FileNotFoundError:
            manifest_revision = "missing"
        return ":".join(
            [
                str(getattr(collection, "points_count", 0)),
                str(getattr(collection, "indexed_vectors_count", 0)),
                manifest_revision,
            ]
        )

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
                    id=str(uuid5(NAMESPACE_URL, doc.id)),
                    vector={TEXT_VECTOR: text_vector, IMAGE_VECTOR: image_vector},
                    payload=doc.model_dump(),
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.settings.collection_name, points=points, wait=True
            )

    def delete_superseded_source_versions(
        self,
        source_id: str,
        source_version: str,
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
                            key="source_version",
                            match=models.MatchValue(value=source_version),
                        )
                    ],
                )
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
        qfilter = _build_filter(request)
        limit = max(request.top_k * 3, request.top_k)
        text_hits = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=text_vector,
            using=TEXT_VECTOR,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        ).points
        image_hits = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=image_vector,
            using=IMAGE_VECTOR,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        ).points
        merged: dict[str, tuple[float, dict, dict]] = {}
        for source, hits in [("text", text_hits), ("image", image_hits)]:
            for hit in hits:
                payload = dict(hit.payload or {})
                lexical = _lexical_bonus(expanded_query, payload.get("text", ""))
                score = float(hit.score) + lexical
                if hit.id not in merged or score > merged[hit.id][0]:
                    merged[hit.id] = (
                        score,
                        payload,
                        {
                            "vector_source": source,
                            "vector_score": float(hit.score),
                            "lexical_bonus": lexical,
                        },
                    )
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
            vector_score = float(debug["vector_score"])
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
        )

    def page_context(
        self,
        source_pdf: str,
        page: int,
        source_version_id: str | None = None,
    ) -> PageContextResponse | None:
        conditions = [
            models.FieldCondition(key="source_pdf", match=models.MatchValue(value=source_pdf)),
            models.FieldCondition(key="page", match=models.MatchValue(value=page)),
        ]
        if source_version_id:
            conditions.append(
                models.FieldCondition(
                    key="source_version_id",
                    match=models.MatchValue(value=source_version_id),
                )
            )
        hits, _ = self.client.scroll(
            collection_name=self.settings.collection_name,
            scroll_filter=models.Filter(must=conditions),
            limit=256,
            with_payload=True,
        )
        payloads = [dict(hit.payload or {}) for hit in hits]
        if not payloads:
            return None

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
            source_pdf=source_pdf,
            team=first.get("team"),
            year=first.get("year"),
            page=page,
            text=text,
            text_chunks=text_chunks,
            page_image_url=page_image_url,
            image_urls=image_urls,
            result_ids=[payload.get("id", "") for payload in payloads if payload.get("id")],
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
        payloads = self._iter_source_payloads(
            _metadata_filter(
                team=team,
                year=year,
                source=source,
                team_numbers=team_numbers,
                years=years,
                source_ids=source_ids,
            )
        )
        summaries = self._summarize_sources(payloads)
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
        payloads = self._iter_source_payloads(_metadata_filter(source=source_pdf))
        summaries = self._summarize_sources(payloads)
        return summaries[0] if summaries else None

    def source_search_from_results(
        self, query: str | None, results: list[SearchResult]
    ) -> list[SourcePageMatch]:
        grouped: dict[tuple[str, int], SourcePageMatch] = {}
        for result in results:
            key = (result.source_pdf, result.page)
            existing = grouped.get(key)
            snippet = result.text[:500] if result.text else ""
            image_urls = [url for url in [result.artifact_url, *result.linked_artifact_urls] if url]
            if existing is None:
                grouped[key] = SourcePageMatch(
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

    def similar_from_result_id(self, result_id: str, top_k: int) -> SimilarPagesResponse | None:
        payload, vectors = self._payload_and_vectors_for_result_id(result_id)
        if payload is None or vectors is None:
            return None
        vector_name = (
            IMAGE_VECTOR
            if payload.get("modality") in {"page_image", "extracted_image"}
            else TEXT_VECTOR
        )
        vector = vectors.get(vector_name)
        if vector is None:
            return None
        return self._similar_from_vector(vector_name, vector, top_k, payload)

    def similar_from_page(
        self, source_pdf: str, page: int, top_k: int
    ) -> SimilarPagesResponse | None:
        payloads = self._scroll_payloads(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_pdf", match=models.MatchValue(value=source_pdf)
                    ),
                    models.FieldCondition(key="page", match=models.MatchValue(value=page)),
                    models.FieldCondition(key="modality", match=models.MatchValue(value="text")),
                ]
            ),
            limit=1,
            with_vectors=True,
        )
        if not payloads:
            return None
        payload, vectors = payloads[0]
        vector = vectors.get(TEXT_VECTOR) if vectors else None
        if vector is None:
            return None
        return self._similar_from_vector(TEXT_VECTOR, vector, top_k, payload)

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
                payloads = self._scroll_payloads(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key="artifact_path", match=models.MatchValue(value=artifact_path)
                            ),
                            models.FieldCondition(
                                key="linked_artifacts", match=models.MatchAny(any=[artifact_path])
                            ),
                        ]
                    ),
                    limit=1,
                )
                payload = payloads[0] if payloads else None
        if payload is None:
            return None
        source_pdf = payload.get("source_pdf", "")
        page = int(payload.get("page", 0))
        source_version_id = payload.get("source_version_id")
        context = self.page_context(source_pdf, page, source_version_id)
        if context is None:
            return None
        return ImageContextResponse(
            result_id=payload.get("id"),
            image_url=self._artifact_url(payload.get("artifact_path")) or image_url,
            source_id=payload.get("source_id"),
            source_version=payload.get("source_version"),
            source_version_id=source_version_id,
            source_pdf=source_pdf,
            team=payload.get("team"),
            year=payload.get("year"),
            page=page,
            page_context_url=self._page_context_url(source_pdf, page, source_version_id),
            page_text_url=self._page_text_url(source_pdf, page, source_version_id),
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
    ) -> SimilarPagesResponse:
        hits = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            using=vector_name,
            limit=max(top_k * 4, top_k + 5),
            with_payload=True,
        ).points
        results: list[SearchResult] = []
        seen_pages: set[tuple[str, int]] = set()
        seed_page = _page_key(seed_payload)
        for hit in hits:
            if float(hit.score) < self.settings.search_min_score:
                continue
            payload = dict(hit.payload or {})
            page_key = _page_key(payload)
            if page_key == seed_page or page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            results.append(
                self._search_result_from_payload(
                    payload, float(hit.score), {"vector_source": vector_name}
                )
            )
            if len(results) >= top_k:
                break
        return SimilarPagesResponse(
            seed={
                "id": seed_payload.get("id"),
                "source_pdf": seed_payload.get("source_pdf"),
                "page": seed_payload.get("page"),
                "modality": seed_payload.get("modality"),
            },
            results=results,
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
                payload.get("source_version_id"),
            ),
            page_text_url=self._page_text_url(
                payload.get("source_pdf", ""),
                int(payload.get("page", 0)),
                payload.get("source_version_id"),
            ),
            debug=debug or {},
        )

    def _payload_and_vectors_for_result_id(self, result_id: str) -> tuple[dict | None, dict | None]:
        points = self.client.retrieve(
            collection_name=self.settings.collection_name,
            ids=[str(uuid5(NAMESPACE_URL, result_id))],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None, None
        point = points[0]
        return dict(point.payload or {}), dict(point.vector or {})

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
        grouped: dict[str, _SourceAccumulator] = {}
        for payload in payloads:
            source_id = str(payload.get("source_id") or Path(payload.get("source_pdf", "")).stem)
            source_version_id = str(payload.get("source_version_id") or source_id)
            accumulator = grouped.get(source_version_id)
            if accumulator is None:
                accumulator = _SourceAccumulator(
                    source_id=source_id,
                    source_version=payload.get("source_version"),
                    source_version_id=source_version_id,
                    source_pdf=payload.get("source_pdf", ""),
                    team=payload.get("team"),
                    year=payload.get("year"),
                    ingested_at=payload.get("ingested_at"),
                    source_url=payload.get("source_url"),
                )
                grouped[source_version_id] = accumulator
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
        prefixes = [self.settings.artifact_url_base.rstrip("/"), "/artifacts"]
        for prefix in prefixes:
            if image_url.startswith(prefix + "/"):
                rel = image_url.removeprefix(prefix + "/")
                return str((self.settings.artifact_dir / rel).resolve())
        return None

    def _artifact_url(self, artifact_path: str | None) -> str | None:
        if not artifact_path:
            return None
        try:
            rel = Path(artifact_path).resolve().relative_to(self.settings.artifact_dir.resolve())
        except ValueError:
            return None
        return f"{self.settings.artifact_url_base.rstrip('/')}/{rel.as_posix()}"

    def _page_context_url(
        self,
        source_pdf: str,
        page: int,
        source_version_id: str | None = None,
    ) -> str:
        path = f"/pages/{quote(source_pdf, safe='')}/{page}"
        return _with_source_version(path, source_version_id)

    def _page_text_url(
        self,
        source_pdf: str,
        page: int,
        source_version_id: str | None = None,
    ) -> str:
        path = f"/pages/{quote(source_pdf, safe='')}/{page}/text"
        return _with_source_version(path, source_version_id)


def _metadata_filter(
    team: str | None = None,
    year: int | None = None,
    source: str | None = None,
    team_numbers: list[str] | None = None,
    years: list[int] | None = None,
    source_ids: list[str] | None = None,
) -> models.Filter | None:
    conditions = []
    if team:
        conditions.append(models.FieldCondition(key="team", match=models.MatchValue(value=team)))
    if team_numbers:
        conditions.append(_match_values("team", list(dict.fromkeys(team_numbers))))
    if year:
        conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))
    if years:
        conditions.append(_match_values("year", list(dict.fromkeys(years))))
    if source:
        conditions.append(
            models.FieldCondition(key="source_pdf", match=models.MatchValue(value=source))
        )
    if source_ids:
        conditions.append(_match_values("source_id", list(dict.fromkeys(source_ids))))
    return models.Filter(must=conditions) if conditions else None


def _build_filter(request: SearchRequest) -> models.Filter | None:
    metadata_filter = _metadata_filter(
        team=request.team,
        year=request.year,
        source=request.source,
        team_numbers=request.team_numbers,
        years=request.years,
        source_ids=request.source_ids,
    )
    conditions = list(metadata_filter.must or []) if metadata_filter else []
    if request.modality:
        conditions.append(
            models.FieldCondition(key="modality", match=models.MatchValue(value=request.modality))
        )
    return models.Filter(must=conditions) if conditions else None


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


def _score_band(score: float, minimum: float) -> ScoreBand:
    if score >= minimum + 0.25:
        return "strong"
    if score >= minimum + 0.1:
        return "good"
    return "relevant"


def _chunk_sort_key(payload: dict) -> tuple[int, str]:
    match = re.search(r"_text_(\d+)$", str(payload.get("id", "")))
    return (int(match.group(1)) if match else 0, str(payload.get("id", "")))


def _page_key(payload: dict) -> tuple[str, int]:
    source = str(payload.get("source_version_id") or payload.get("source_pdf", ""))
    return source, int(payload.get("page", 0))


def _with_source_version(path: str, source_version_id: str | None) -> str:
    if not source_version_id:
        return path
    return f"{path}?source_version_id={quote(source_version_id, safe='')}"
