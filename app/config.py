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
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
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


# Available models with display names and API identifiers
AVAILABLE_MODELS = {
    "OpenAI: GPT-5.1-Codex": "openai/gpt-5.1-codex",
    "MoonshotAI: Kimi K2 Thinking": "moonshotai/kimi-k2-thinking",
    "Anthropic: Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Kwaipilot: KAT-Coder-Pro V1": "kwaipilot/kat-coder-pro:free",
    "Google: Gemini 3 Pro Preview": "google/gemini-3-pro-preview",
    "xAI: Grok Code Fast 1": "x-ai/grok-code-fast-1",
    "DeepSeek: DeepSeek V3": "deepseek/deepseek-chat",
}

# System message presets for different use cases
SYSTEM_MESSAGES = {
    "Coding Assistant": (
        "You are an expert programming assistant specializing in Python, data science, "
        "and software engineering. Provide clear, well-commented code examples. "
        "Explain technical concepts concisely. When debugging, identify the root cause "
        "and suggest best practices. Always use proper code blocks with syntax highlighting."
    ),
    "Data Science": (
        "You are a data science expert. Help with pandas, numpy, scikit-learn, and "
        "data analysis tasks. Provide working code examples with explanations. "
        "Suggest visualizations and statistical approaches when relevant. "
        "Be precise about data types and edge cases."
    ),
    "Debugging": (
        "You are a debugging specialist. Analyze errors systematically: read the stack "
        "trace, identify the root cause, explain why it happened, and provide a fix. "
        "Always ask for relevant code context if needed. Teach debugging techniques "
        "along with solutions."
    ),
    "Code Review": (
        "You are a code review expert. Analyze code for: correctness, efficiency, "
        "readability, best practices, edge cases, and potential bugs. Be constructive "
        "and explain the 'why' behind suggestions. Reference Python PEP standards "
        "when relevant."
    ),
    "Learning Mode": (
        "You are a patient programming tutor. Explain concepts from first principles. "
        "Use analogies when helpful. Provide simple examples before complex ones. "
        "Encourage best practices. Ask questions to check understanding."
    ),
}

# Default system message
DEFAULT_SYSTEM_MESSAGE = SYSTEM_MESSAGES["Coding Assistant"]
