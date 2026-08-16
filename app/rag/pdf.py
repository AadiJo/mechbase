import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from app.rag.artifacts import generation_namespace, source_artifact_root
from app.rag.chunking import (
    MECHANISM_HEADINGS,
    resolve_page_section,
    section_candidates,
    split_text,
)
from app.rag.config import Settings
from app.rag.models import RagDocument, SourceDoc


def _document_id(
    source_version_id: str,
    page: int,
    modality: str,
    index: int | None = None,
) -> str:
    identity = json.dumps(
        [source_version_id, page, modality, index],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return f"doc_{sha256(identity).hexdigest()}"


def _storage_id(document_id: str, ingestion_id: str | None) -> str:
    return f"{document_id}@{ingestion_id}" if ingestion_id else document_id


def _render_page(page: fitz.Page, dpi: int, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    pix.save(out_path)
    return out_path


def _ocr_image(path: Path) -> str:
    with Image.open(path) as image:
        return pytesseract.image_to_string(image).strip()


def extract_documents(
    source: SourceDoc,
    settings: Settings,
    *,
    ingestion_id: str | None = None,
    artifact_namespace: str | None = None,
) -> list[RagDocument]:
    document_namespace = artifact_namespace or generation_namespace(
        source.source_version, ingestion_id
    )
    artifact_root = (
        source_artifact_root(settings.artifact_dir, source.source_id) / document_namespace
    )
    docs: list[RagDocument] = []
    pdf = fitz.open(source.path)
    try:
        outline_sections = _outline_section_starts(pdf)
        repeated_headers = _repeated_headers(pdf)
        inherited_section = None
        for page_index, page in enumerate(pdf):
            page_num = page_index + 1
            page_dir = artifact_root / f"page-{page_num:03d}"
            native_text = page.get_text("text").strip()
            page_image_path = page_dir / "page.png"
            if len(native_text) < settings.ocr_min_chars_per_page:
                _render_page(page, settings.render_dpi, page_image_path)
                ocr_text = _ocr_image(page_image_path)
                page_text = "\n\n".join(part for part in [native_text, ocr_text] if part).strip()
            else:
                page_text = native_text
                if not page_image_path.exists():
                    _render_page(page, settings.render_dpi, page_image_path)

            section = resolve_page_section(
                page_text,
                inherited_section,
                repeated_headers,
                outline_heading=outline_sections.get(page_num),
            )
            inherited_section = section
            linked_artifacts = [str(page_image_path)]
            extracted_images = _extract_page_images(
                pdf,
                page,
                page_dir,
                source,
                page_num,
                page_text,
                section,
                ingestion_id,
            )
            linked_artifacts.extend(
                doc.artifact_path for doc in extracted_images if doc.artifact_path
            )
            docs.extend(extracted_images)

            page_document_id = _document_id(source.source_version_id, page_num, "page")
            docs.append(
                RagDocument(
                    id=page_document_id,
                    storage_id=_storage_id(page_document_id, ingestion_id),
                    source_id=source.source_id,
                    source_version=source.source_version,
                    source_version_id=source.source_version_id,
                    ingestion_id=ingestion_id,
                    is_staged=ingestion_id is not None,
                    source_pdf=source.path.name,
                    team=source.team,
                    year=source.year,
                    page=page_num,
                    modality="page_image",
                    text=page_text[:3000],
                    artifact_path=str(page_image_path),
                    linked_artifacts=linked_artifacts,
                    section=section,
                    source_url=source.source_url,
                )
            )
            for chunk_idx, chunk in enumerate(
                split_text(page_text, settings.chunk_target_chars, settings.chunk_overlap_chars)
            ):
                text_document_id = _document_id(
                    source.source_version_id,
                    page_num,
                    "text",
                    chunk_idx,
                )
                docs.append(
                    RagDocument(
                        id=text_document_id,
                        storage_id=_storage_id(text_document_id, ingestion_id),
                        source_id=source.source_id,
                        source_version=source.source_version,
                        source_version_id=source.source_version_id,
                        ingestion_id=ingestion_id,
                        is_staged=ingestion_id is not None,
                        source_pdf=source.path.name,
                        team=source.team,
                        year=source.year,
                        page=page_num,
                        modality="text",
                        chunk_index=chunk_idx,
                        text=chunk,
                        linked_artifacts=linked_artifacts,
                        section=section,
                        source_url=source.source_url,
                    )
                )
    finally:
        pdf.close()
    return docs


def _extract_page_images(
    pdf: fitz.Document,
    page: fitz.Page,
    page_dir: Path,
    source: SourceDoc,
    page_num: int,
    page_text: str,
    section: str | None,
    ingestion_id: str | None,
) -> list[RagDocument]:
    docs: list[RagDocument] = []
    seen: set[int] = set()
    for image_idx, image_info in enumerate(page.get_images(full=True)):
        xref = image_info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            image = pdf.extract_image(xref)
        except Exception:
            continue
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width < 120 or height < 120:
            continue
        ext = image.get("ext", "png")
        out_path = page_dir / f"image-{image_idx:03d}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            out_path.write_bytes(image["image"])
        image_document_id = _document_id(
            source.source_version_id,
            page_num,
            "image",
            image_idx,
        )
        docs.append(
            RagDocument(
                id=image_document_id,
                storage_id=_storage_id(image_document_id, ingestion_id),
                source_id=source.source_id,
                source_version=source.source_version,
                source_version_id=source.source_version_id,
                ingestion_id=ingestion_id,
                is_staged=ingestion_id is not None,
                source_pdf=source.path.name,
                team=source.team,
                year=source.year,
                page=page_num,
                modality="extracted_image",
                text=page_text[:1500],
                artifact_path=str(out_path),
                linked_artifacts=[str(out_path)],
                section=section,
                source_url=source.source_url,
            )
        )
    return docs


def _outline_section_starts(pdf: fitz.Document) -> dict[int, str]:
    try:
        outline = pdf.get_toc(simple=True)
    except (RuntimeError, ValueError):
        return {}
    starts = [
        (int(page), " ".join(str(title).split()))
        for _level, title, page in outline
        if str(title).strip() and 1 <= int(page) <= len(pdf)
    ]
    if not starts:
        return {}

    starts_by_page: dict[int, list[str]] = {}
    for page, title in starts:
        starts_by_page.setdefault(page, []).append(title)
    return {page: titles[-1] for page, titles in starts_by_page.items()}


def _repeated_headers(pdf: fitz.Document) -> set[str]:
    candidate_counts: Counter[str] = Counter()
    for page in pdf:
        candidates = [
            candidate.casefold() for candidate in section_candidates(page.get_text("text"))[:2]
        ]
        candidate_counts.update(set(candidates))
    minimum_repeats = max(2, math.ceil(len(pdf) * 0.6))
    repeated = {
        candidate for candidate, count in candidate_counts.items() if count >= minimum_repeats
    }
    repeated = {
        candidate
        for candidate in repeated
        if not any(f" {mechanism} " in f" {candidate} " for mechanism in MECHANISM_HEADINGS)
    }
    if not repeated:
        return set()

    alternatives = set(candidate_counts) - repeated
    if alternatives:
        return repeated

    # Repeated heading hierarchies are ambiguous without layout information. Suppress only clear
    # document chrome and retain headings such as "Swerve Drive / Mechanical Design" intact.
    document_markers = (" binder", " report", "team ", "frc ", "technical ")
    return {
        candidate
        for candidate in repeated
        if any(marker in f" {candidate}" for marker in document_markers)
    }
