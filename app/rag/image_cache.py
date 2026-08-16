import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps


def cached_resized_image(source: Path, max_side: int, cache_dir: Path) -> Path:
    digest = sha256()
    with source.open("rb") as image_file:
        while chunk := image_file.read(1024 * 1024):
            digest.update(chunk)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{digest.hexdigest()}-{max_side}.jpg"
    if _is_valid_jpeg(cached):
        return cached

    temporary = cache_dir / f".{cached.name}.{uuid4().hex}.tmp"
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
            rgb.save(temporary, "JPEG", quality=82, optimize=True)
        with temporary.open("rb") as output:
            os.fsync(output.fileno())
        temporary.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)
    return cached


def _is_valid_jpeg(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
            return image.format == "JPEG"
    except OSError:
        return False
