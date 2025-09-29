from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BACKEND_", extra="allow")

    environment: str = "development"
    log_level: str = "INFO"
    postgres_dsn: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/spotify"
    redis_url: str = "redis://redis:6379/0"
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    service_token: str = ""
    faiss_index_path: Path = Path("/data/index.faiss")
    faiss_meta_path: Path = Path("/data/index.json")
    dsp_preview_timeout: int = 12
    recommendation_top_k: int = 15
    playlist_ingest_batch_size: int = 75
    http_timeout_seconds: float = 15.0
    http_retries: int = 3
    allow_origins: List[str] = ["*"]
    jwt_issuer: str = "spotify-rec-backend"
    jwt_audience: str = "spotify-rec-frontend"
    service_token_algorithm: str = "HS256"

    @field_validator("faiss_index_path", "faiss_meta_path", mode="before")
    @classmethod
    def _expand_paths(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()