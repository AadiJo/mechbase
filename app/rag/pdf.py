from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from app.rag.chunking import section_from_text, split_text
from app.rag.config import Settings
from app.rag.models import RagDocument, SourceDoc


def _safe_id(*parts: object) -> str:
    return "_".join(str(part).replace("/", "_").replace(" ", "_") for part in parts)


def _render_page(page: fitz.Page, dpi: int, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    pix.save(out_path)
    return out_path


def _ocr_image(path: Path) -> str:
    with Image.open(path) as image:
        return pytesseract.image_to_string(image).strip()


def extract_documents(source: SourceDoc, settings: Settings) -> list[RagDocument]:
    artifact_root = settings.artifact_dir / source.source_id
    docs: list[RagDocument] = []
    pdf = fitz.open(source.path)
    try:
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

            linked_artifacts = [str(page_image_path)]
            extracted_images = _extract_page_images(pdf, page, page_dir, source, page_num, page_text)
            linked_artifacts.extend(doc.artifact_path for doc in extracted_images if doc.artifact_path)
            docs.extend(extracted_images)

            section = section_from_text(page_text)
            docs.append(
                RagDocument(
                    id=_safe_id(source.source_id, page_num, "page"),
                    source_id=source.source_id,
                    source_pdf=source.path.name,
                    team=source.team,
                    year=source.year,
                    page=page_num,
                    modality="page_image",
                    text=page_text[:3000],
                    artifact_path=str(page_image_path),
                    linked_artifacts=linked_artifacts,
                    section=section,
                )
            )
            for chunk_idx, chunk in enumerate(
                split_text(page_text, settings.chunk_target_chars, settings.chunk_overlap_chars)
            ):
                docs.append(
                    RagDocument(
                        id=_safe_id(source.source_id, page_num, "text", chunk_idx),
                        source_id=source.source_id,
                        source_pdf=source.path.name,
                        team=source.team,
                        year=source.year,
                        page=page_num,
                        modality="text",
                        text=chunk,
                        linked_artifacts=linked_artifacts,
                        section=section,
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
        docs.append(
            RagDocument(
                id=_safe_id(source.source_id, page_num, "image", image_idx),
                source_id=source.source_id,
                source_pdf=source.path.name,
                team=source.team,
                year=source.year,
                page=page_num,
                modality="extracted_image",
                text=page_text[:1500],
                artifact_path=str(out_path),
                linked_artifacts=[str(out_path)],
                section=section_from_text(page_text),
            )
        )
    return docs
