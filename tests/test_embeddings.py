from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agentic_rag_knowledge_system.embeddings import (
    FakeEmbeddingModel,
    Qwen3EmbeddingModel,
    build_vector_records,
    format_qwen3_query,
    stable_vector_id,
)
from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.pipeline import ingest_text_directory
from agentic_rag_knowledge_system.settings import get_settings

EXAMPLE_CORPUS = Path(__file__).parents[1] / "examples" / "corpus"


def test_fake_embeddings_are_deterministic_and_dimensioned() -> None:
    model = FakeEmbeddingModel(dimension=6)

    first = model.embed_texts(("same text",))[0]
    second = model.embed_texts(("same text",))[0]

    assert first == second
    assert len(first) == 6


def test_qwen_query_instruction_format() -> None:
    formatted = format_qwen3_query("What is RAG?")

    assert formatted.startswith("Instruct: ")
    assert formatted.endswith("Query: What is RAG?")


def test_build_vector_records_preserves_chunk_lineage() -> None:
    ingestion = ingest_text_directory(
        EXAMPLE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
    )
    model = FakeEmbeddingModel(dimension=4)

    records = build_vector_records(ingestion.chunks[:2], model=model)

    assert len(records) == 2
    assert records[0].chunk_id == ingestion.chunks[0].chunk_id
    assert records[0].source_uri == "llm-playbook.md"
    assert records[0].dimension == 4
    assert records[0].vector_id == stable_vector_id(
        chunk_id=ingestion.chunks[0].chunk_id,
        model_name=model.model_name,
    )


def test_build_vector_records_rejects_dimension_mismatch() -> None:
    class BadModel:
        model_name = "bad"
        dimension = 3

        def embed_texts(self, texts):
            return ((1.0, 2.0),)

        def embed_query(self, query):
            return (1.0, 2.0)

    ingestion = ingest_text_directory(EXAMPLE_CORPUS)

    with pytest.raises(ValueError, match="embedding dimension"):
        build_vector_records(ingestion.chunks[:1], model=BadModel())


def test_qwen_adapter_passes_hf_token_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class StubSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

        def encode(self, texts, **kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=StubSentenceTransformer),
    )
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    get_settings.cache_clear()

    Qwen3EmbeddingModel(dimension=2)

    assert captured["kwargs"] == {"token": "hf_test_token"}
    get_settings.cache_clear()


@pytest.mark.skip(reason="Real Qwen model smoke test; run manually from the notebook.")
def test_qwen3_embedding_model_smoke() -> None:
    model = Qwen3EmbeddingModel(dimension=1024)
    query_vector = model.embed_query("What is prompt caching?")

    assert len(query_vector) == 1024
