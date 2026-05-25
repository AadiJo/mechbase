from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACT_DIR")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    collection_name: str = Field(default="frc_mechanisms", alias="COLLECTION_NAME")
    artifact_url_base: str = Field(default="/images", alias="ARTIFACT_URL_BASE")
    text_model: str = Field(default="voyage-4", alias="TEXT_MODEL")
    multimodal_model: str = Field(default="voyage-multimodal-3.5", alias="MULTIMODAL_MODEL")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    ocr_min_chars_per_page: int = Field(default=40, alias="OCR_MIN_CHARS_PER_PAGE")
    render_dpi: int = Field(default=144, alias="RENDER_DPI")
    chunk_target_chars: int = Field(default=1300, alias="CHUNK_TARGET_CHARS")
    chunk_overlap_chars: int = Field(default=220, alias="CHUNK_OVERLAP_CHARS")
    multimodal_batch_size: int = Field(default=4, alias="MULTIMODAL_BATCH_SIZE")
    max_embed_image_side: int = Field(default=1400, alias="MAX_EMBED_IMAGE_SIDE")
    required_api_key_permission: str = Field(default="search:read", alias="REQUIRED_API_KEY_PERMISSION")
    convex_http_url: str | None = Field(default=None, alias="CONVEX_HTTP_URL")
    convex_recording_secret: str | None = Field(default=None, alias="CONVEX_RECORDING_SECRET")


@lru_cache
def get_settings() -> Settings:
    return Settings()
