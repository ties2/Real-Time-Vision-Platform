from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BatchingConfig(BaseModel):
    """Dynamic batching settings.

    Override from the environment with a double underscore, e.g.:
        BATCHING__ENABLED=false
        BATCHING__MAX_BATCH_SIZE=8
        BATCHING__MAX_WAIT_MS=5
    """

    enabled: bool = True
    max_batch_size: int = 4
    max_wait_ms: int = 10


class Settings(BaseSettings):
    """Application configuration."""

    app_name: str = "Real-Time Vision Platform"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8000

    log_level: str = "INFO"

    model_registry_path: Path = Field(default=PROJECT_ROOT / "models")

    models_config_path: Path = Field(default=PROJECT_ROOT / "configs" / "models.yaml")
    default_model: str = "yolo11"

    # model_confidence_threshold: float = 0.5

    max_image_size_mb: int = 10

    api_prefix: str = "/api/v1"
    batching: BatchingConfig = BatchingConfig()
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()