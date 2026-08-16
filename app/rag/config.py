from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        url_preserve_empty_path=True,
    )

    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACT_DIR")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    collection_name: str = Field(default="frc_mechanisms", alias="COLLECTION_NAME")
    artifact_url_base: str = Field(default="/images", alias="ARTIFACT_URL_BASE")
    text_model: str = Field(default="voyage-4", alias="TEXT_MODEL")
    multimodal_model: str = Field(default="voyage-multimodal-3.5", alias="MULTIMODAL_MODEL")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    search_min_score: float = Field(default=0.35, ge=-1.0, le=1.0, alias="SEARCH_MIN_SCORE")
    ocr_min_chars_per_page: int = Field(default=40, alias="OCR_MIN_CHARS_PER_PAGE")
    render_dpi: int = Field(default=144, alias="RENDER_DPI")
    chunk_target_chars: int = Field(default=1300, alias="CHUNK_TARGET_CHARS")
    chunk_overlap_chars: int = Field(default=220, alias="CHUNK_OVERLAP_CHARS")
    multimodal_batch_size: int = Field(default=4, alias="MULTIMODAL_BATCH_SIZE")
    max_embed_image_side: int = Field(default=1400, alias="MAX_EMBED_IMAGE_SIDE")
    required_api_key_permission: str = Field(
        default="search:read", alias="REQUIRED_API_KEY_PERMISSION"
    )
    convex_http_url: str | None = Field(default=None, alias="CONVEX_HTTP_URL")
    convex_recording_secret: str | None = Field(default=None, alias="CONVEX_RECORDING_SECRET")
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_max_requests: int = Field(default=20, alias="RATE_LIMIT_MAX_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    mcp_public_base_url: AnyHttpUrl = Field(
        default="https://api-frcrag-v2.johari-dev.com",
        alias="MCP_PUBLIC_BASE_URL",
    )
    mcp_search_top_k: int = Field(default=10, ge=1, le=20, alias="MCP_SEARCH_TOP_K")
    mcp_oauth_scopes: str = Field(default="openid", alias="MCP_OAUTH_SCOPES")
    mcp_allowed_origins: str = Field(
        default="https://chatgpt.com,https://claude.ai",
        alias="MCP_ALLOWED_ORIGINS",
    )
    clerk_oauth_issuer_url: AnyHttpUrl = Field(
        default="https://clerk.mechbase.johari-dev.com",
        alias="CLERK_OAUTH_ISSUER_URL",
    )
    clerk_secret_key: str | None = Field(default=None, alias="CLERK_SECRET_KEY")

    @property
    def mcp_endpoint_url(self) -> str:
        return f"{str(self.mcp_public_base_url).rstrip('/')}/mcp"

    @property
    def mcp_required_scopes(self) -> list[str]:
        return self.mcp_oauth_scopes.split()

    @property
    def mcp_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.mcp_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
