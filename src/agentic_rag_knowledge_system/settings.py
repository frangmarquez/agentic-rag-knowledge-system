"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppSettings(BaseSettings):
    """Typed application settings.

    Environment variables use the AGENTIC_RAG_ prefix, for example
    AGENTIC_RAG_LOG_LEVEL=DEBUG.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTIC_RAG_",
        extra="ignore",
    )

    app_name: str = "agentic-rag-knowledge-system"
    environment: Environment = "local"
    log_level: LogLevel = "INFO"
    debug: bool = False
    api_prefix: str = "/api/v1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "llm_playbook_qwen3"
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "AGENTIC_RAG_HF_TOKEN"),
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return str(value).upper()

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("api_prefix must start with '/'")

        normalized = value.rstrip("/")
        return normalized or "/"

    @field_validator("qdrant_url", "qdrant_collection")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings for runtime use."""

    return AppSettings()
