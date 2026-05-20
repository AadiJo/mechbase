import base64
import mimetypes
import time
from pathlib import Path

import httpx
from PIL import Image

from app.rag.config import Settings


class MissingVoyageApiKey(RuntimeError):
    pass


class VoyageEmbedder:
    def __init__(self, settings: Settings):
        if not settings.voyage_api_key:
            raise MissingVoyageApiKey("VOYAGE_API_KEY is required for ingestion and search.")
        self.settings = settings
        self.client = httpx.Client(
            base_url="https://api.voyageai.com/v1",
            headers={"Authorization": f"Bearer {settings.voyage_api_key}"},
            timeout=httpx.Timeout(180, connect=20),
        )

    def embed_texts(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(
            "/embeddings",
            {
                "input": texts,
                "model": self.settings.text_model,
                "input_type": input_type,
                "output_dimension": self.settings.embedding_dim,
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [list(item["embedding"]) for item in data]

    def embed_multimodal(
        self, texts: list[str], image_paths: list[str | None], input_type: str
    ) -> list[list[float]]:
        if not texts:
            return []
        inputs = []
        for text, image_path in zip(texts, image_paths, strict=True):
            content: list[dict] = []
            if text:
                content.append({"type": "text", "text": text})
            if image_path:
                content.append(
                    {
                        "type": "image_base64",
                        "image_base64": _data_url(
                            _embedding_image(Path(image_path), self.settings.max_embed_image_side)
                        ),
                    }
                )
            inputs.append({"content": content or [{"type": "text", "text": "FRC robot mechanism image"}]})
        response = self._post(
            "/multimodalembeddings",
            {
                "inputs": inputs,
                "model": self.settings.multimodal_model,
                "input_type": input_type,
                "output_dimension": self.settings.embedding_dim,
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [list(item["embedding"]) for item in data]

    def _post(self, path: str, payload: dict) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.post(path, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(2**attempt)
                    continue
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                time.sleep(2**attempt)
        if last_exc:
            raise last_exc
        return response


def _data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _embedding_image(path: Path, max_side: int) -> Path:
    cache_dir = path.parent / ".embed"
    cache_dir.mkdir(exist_ok=True)
    out = cache_dir / f"{path.stem}-{max_side}.jpg"
    if out.exists():
        return out
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        image.save(out, "JPEG", quality=82, optimize=True)
    return out
