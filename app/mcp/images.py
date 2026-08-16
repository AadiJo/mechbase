from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.rag.config import Settings
from app.rag.image_cache import cached_resized_image
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

    try:
        cached = cached_resized_image(
            source,
            max_side,
            settings.artifact_dir / ".embedding-cache",
        )
        data = cached.read_bytes()
    except OSError:
        return None
    return PreviewImage(data=data, mime_type="image/jpeg")


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
