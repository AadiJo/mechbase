from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError

from app.rag.config import Settings
from app.rag.image_cache import cached_resized_image
from app.rag.models import ImageContextResponse


@dataclass(frozen=True, slots=True)
class PreviewImage:
    data: bytes
    mime_type: str
    source_mime_type: str
    width: int
    height: int
    preview_width: int
    preview_height: int


@dataclass(frozen=True, slots=True)
class CandidateAssetSource:
    asset_id: str
    kind: Literal["page", "figure"]
    image_url: str


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

    return load_preview_url(image_url, settings, max_side=max_side)


def load_preview_url(
    image_url: str,
    settings: Settings,
    *,
    max_side: int = 1400,
) -> PreviewImage | None:
    source = _artifact_path_from_url(image_url, settings)
    if source is None or not source.is_file():
        return None

    try:
        with Image.open(source) as original:
            width, height = original.size
            source_mime_type = Image.MIME.get(
                original.format or "",
                "application/octet-stream",
            )
        cached = cached_resized_image(
            source,
            max_side,
            settings.artifact_dir / ".embedding-cache",
        )
        data = cached.read_bytes()
        with Image.open(cached) as resized:
            preview_width, preview_height = resized.size
    except (OSError, Image.DecompressionBombError, UnidentifiedImageError):
        return None
    return PreviewImage(
        data=data,
        mime_type="image/jpeg",
        source_mime_type=source_mime_type,
        width=width,
        height=height,
        preview_width=preview_width,
        preview_height=preview_height,
    )


def candidate_asset_sources(
    context: ImageContextResponse,
    *,
    include_assets: bool,
    max_figures: int = 4,
) -> list[CandidateAssetSource]:
    page_image_url = context.page_image_url
    assets = []
    seen_urls = set()
    seen_asset_ids = set()
    if page_image_url:
        assets.append(
            CandidateAssetSource(
                asset_id=_asset_id_from_url("page", page_image_url),
                kind="page",
                image_url=page_image_url,
            )
        )
        seen_urls.add(page_image_url)
        seen_asset_ids.add(assets[0].asset_id)

    candidate_urls = [context.image_url, *context.image_urls]
    figures = []
    for image_url in candidate_urls:
        if image_url is None or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        asset_id = _asset_id_from_url("figure", image_url)
        if asset_id in seen_asset_ids:
            continue
        seen_asset_ids.add(asset_id)
        figures.append(
            CandidateAssetSource(
                asset_id=asset_id,
                kind="figure",
                image_url=image_url,
            )
        )
        if len(figures) == max_figures:
            break

    if include_assets:
        return [*assets, *figures]
    if assets:
        return assets[:1]
    return figures[:1]


def _asset_id_from_url(kind: Literal["page", "figure"], image_url: str) -> str:
    artifact_path = unquote(urlparse(image_url).path)
    digest = sha256(artifact_path.encode()).hexdigest()[:20]
    return f"{kind}-{digest}"


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
