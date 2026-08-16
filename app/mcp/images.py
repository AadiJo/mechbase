from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from app.rag.config import Settings
from app.rag.models import ImageContextResponse


@dataclass(frozen=True, slots=True)
class PreviewImage:
    data: bytes
    mime_type: str


def preview_image_url(context: ImageContextResponse) -> str | None:
    return context.page_image_url or context.image_url or next(iter(context.image_urls), None)


def load_preview_image(
    context: ImageContextResponse,
    settings: Settings,
    *,
    max_side: int = 1400,
) -> PreviewImage | None:
    image_url = preview_image_url(context)
    if image_url is None:
        return None

    source = _artifact_path_from_url(image_url, settings)
    if source is None or not source.is_file():
        return None

    cached_preview = source.parent / ".embed" / f"{source.stem}-{max_side}.jpg"
    if cached_preview.is_file():
        return PreviewImage(data=cached_preview.read_bytes(), mime_type="image/jpeg")

    try:
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original)
            image.thumbnail((max_side, max_side))
            if image.mode == "RGB":
                rgb = image
            elif "A" in image.getbands():
                rgb = Image.new("RGB", image.size, "white")
                rgb.paste(image, mask=image.getchannel("A"))
            else:
                rgb = image.convert("RGB")

            output = BytesIO()
            rgb.save(output, "JPEG", quality=82, optimize=True)
    except (OSError, UnidentifiedImageError):
        return None

    return PreviewImage(data=output.getvalue(), mime_type="image/jpeg")


def _artifact_path_from_url(image_url: str, settings: Settings) -> Path | None:
    url_path = unquote(urlparse(image_url).path)
    prefixes = {settings.artifact_url_base.rstrip("/"), "/artifacts"}
    relative_path = next(
        (
            url_path.removeprefix(prefix + "/")
            for prefix in prefixes
            if prefix and url_path.startswith(prefix + "/")
        ),
        None,
    )
    if relative_path is None:
        return None

    artifact_root = settings.artifact_dir.resolve()
    candidate = (artifact_root / relative_path).resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError:
        return None
    return candidate
