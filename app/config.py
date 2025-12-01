"""Application configuration helpers."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings  # ← Note: Using BaseSettings


class Settings(BaseSettings):  # ← Changed from BaseModel to BaseSettings
    """Container for strongly-typed application settings."""

    app_name: str = Field(default="LLM Chat Service", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    secret_key: str = Field(default="changeme", alias="SECRET_KEY")
    gradio_mount_path: str = Field(default="/ui", alias="GRADIO_MOUNT_PATH")
    default_model_name: str = Field(
        default="openrouter/gpt-4", alias="DEFAULT_MODEL_NAME"
    )
    cors_origins: List[str] = Field(default_factory=list, alias="CORS_ORIGINS")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/chatdb",
        alias="DATABASE_URL",
    )

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    model_config = ConfigDict(
        populate_by_name=True,
        case_sensitive=False,
        # Only load .env when running locally (not in Docker)
        env_file='.env' if not os.getenv('DOCKER_CONTAINER') else None,
        env_file_encoding='utf-8'
    )

    @property
    def async_database_url(self) -> str:
        """Ensure the SQLAlchemy URL uses an async driver."""
        if "+asyncpg" in self.database_url:
            return self.database_url
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    return settings


def _parse_csv(raw: str) -> List[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]
