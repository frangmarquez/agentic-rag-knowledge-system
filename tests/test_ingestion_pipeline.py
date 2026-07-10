from pathlib import Path

from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.cache import load_or_create_ingestion_cache
from agentic_rag_knowledge_system.ingestion.models import DocumentChunk, SourceDocument
from agentic_rag_knowledge_system.ingestion.pipeline import ingest_text_directory
from agentic_rag_knowledge_system.ingestion.serialization import (
    records_from_jsonl,
    records_to_jsonl,
)

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
EXAMPLE_CORPUS = Path(__file__).parents[1] / "examples" / "corpus"


def test_ingest_text_directory_returns_stable_counts_and_order() -> None:
    result = ingest_text_directory(
        FIXTURE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=0),
    )

    assert result.document_count == 4
    assert [document.source_uri for document in result.documents] == [
        "a.md",
        "b.txt",
        "guide.md",
        "nested/c.markdown",
    ]
    assert result.chunk_count == len(result.chunks)
    assert [chunk.source_uri for chunk in result.chunks][:4] == [
        "a.md",
        "b.txt",
        "guide.md",
        "guide.md",
    ]

    repeated = ingest_text_directory(
        FIXTURE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=0),
    )
    assert [chunk.chunk_id for chunk in result.chunks] == [
        chunk.chunk_id for chunk in repeated.chunks
    ]


def test_ingest_example_llm_playbook_corpus() -> None:
    result = ingest_text_directory(
        EXAMPLE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
        metadata={"corpus": "llm-playbook"},
    )

    assert result.document_count == 1
    assert result.documents[0].source_uri == "llm-playbook.md"
    assert result.documents[0].title == "AI Engineering Handbook"
    assert result.documents[0].metadata["corpus"] == "llm-playbook"
    assert result.chunk_count > 1
    assert {chunk.metadata["corpus"] for chunk in result.chunks} == {"llm-playbook"}


def test_documents_round_trip_through_jsonl() -> None:
    result = ingest_text_directory(FIXTURE_CORPUS)

    payload = records_to_jsonl(result.documents)
    restored = records_from_jsonl(payload, SourceDocument)

    assert restored == list(result.documents)


def test_chunks_round_trip_through_jsonl() -> None:
    result = ingest_text_directory(
        FIXTURE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=20, chunk_overlap_chars=5),
    )

    payload = records_to_jsonl(result.chunks)
    restored = records_from_jsonl(payload, DocumentChunk)

    assert restored == list(result.chunks)


def test_ingestion_cache_reuses_jsonl_records() -> None:
    cache_dir = Path("data/processed/test_ingestion_cache")

    created = load_or_create_ingestion_cache(
        FIXTURE_CORPUS,
        cache_dir,
        chunking_config=ChunkingConfig(chunk_size_chars=20, chunk_overlap_chars=5),
    )
    cached = load_or_create_ingestion_cache(
        Path("missing-corpus"),
        cache_dir,
        chunking_config=ChunkingConfig(chunk_size_chars=20, chunk_overlap_chars=5),
    )

    assert (cache_dir / "documents.jsonl").exists()
    assert (cache_dir / "chunks.jsonl").exists()
    assert cached == created
