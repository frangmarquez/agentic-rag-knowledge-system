import os

import pytest
from pydantic import ValidationError

from agentic_rag_knowledge_system.settings import AppSettings, get_settings

ENV_PREFIX = "AGENTIC_RAG_"
SECRET_ENV_KEYS = ("HF_TOKEN", "AGENTIC_RAG_HF_TOKEN")


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch):
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    for key in SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_defaults() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.app_name == "agentic-rag-knowledge-system"
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.debug is False
    assert settings.api_prefix == "/api/v1"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "llm_playbook_qwen3"
    assert settings.hf_token is None


def test_settings_read_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_RAG_APP_NAME", "custom-rag")
    monkeypatch.setenv("AGENTIC_RAG_ENVIRONMENT", "TEST")
    monkeypatch.setenv("AGENTIC_RAG_LOG_LEVEL", "debug")
    monkeypatch.setenv("AGENTIC_RAG_DEBUG", "true")
    monkeypatch.setenv("AGENTIC_RAG_API_PREFIX", "/internal/")
    monkeypatch.setenv("AGENTIC_RAG_QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("AGENTIC_RAG_QDRANT_COLLECTION", "custom_collection")

    settings = AppSettings(_env_file=None)

    assert settings.app_name == "custom-rag"
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.debug is True
    assert settings.api_prefix == "/internal"
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_collection == "custom_collection"


def test_settings_reads_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    settings = AppSettings(_env_file=None)

    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "hf_test_token"


def test_settings_reject_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_RAG_ENVIRONMENT", "sandbox")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_settings_reject_invalid_api_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_RAG_API_PREFIX", "api/v1")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_RAG_APP_NAME", "cached-rag")

    first = get_settings()
    second = get_settings()

    assert first is second
    assert first.app_name == "cached-rag"
