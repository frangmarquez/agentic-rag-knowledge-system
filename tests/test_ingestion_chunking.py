import pytest
from pydantic import ValidationError

from agentic_rag_knowledge_system.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
)
from agentic_rag_knowledge_system.ingestion.models import (
    SourceDocument,
    compute_content_hash,
    stable_source_id,
)


def make_source_document(content: str) -> SourceDocument:
    source_uri = "docs/test.md"
    return SourceDocument(
        source_id=stable_source_id(source_uri),
        source_uri=source_uri,
        title="Test",
        content=content,
        content_hash=compute_content_hash(content),
        metadata={"domain": "tests"},
    )


def test_chunk_document_uses_stable_overlapping_character_windows() -> None:
    document = make_source_document("abcdefghijklmnopqrstuvwxyz")
    config = ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=2)

    chunks = chunk_document(document, config=config)

    assert [(chunk.start_char, chunk.end_char, chunk.text) for chunk in chunks] == [
        (0, 10, "abcdefghij"),
        (8, 18, "ijklmnopqr"),
        (16, 26, "qrstuvwxyz"),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

    repeated_chunks = chunk_document(document, config=config)
    assert [chunk.chunk_id for chunk in chunks] == [
        chunk.chunk_id for chunk in repeated_chunks
    ]


def test_chunk_document_preserves_lineage_and_metadata() -> None:
    document = make_source_document("chunk me once")

    [chunk] = chunk_document(
        document,
        config=ChunkingConfig(chunk_size_chars=50, chunk_overlap_chars=0),
    )

    assert chunk.source_id == document.source_id
    assert chunk.source_uri == document.source_uri
    assert chunk.metadata == document.metadata
    assert chunk.content_hash == compute_content_hash(chunk.text)


def test_chunk_document_uses_new_chunk_ids_when_content_changes() -> None:
    config = ChunkingConfig(chunk_size_chars=50, chunk_overlap_chars=0)
    first_document = make_source_document("first version")
    second_document = make_source_document("second version")

    [first_chunk] = chunk_document(first_document, config=config)
    [second_chunk] = chunk_document(second_document, config=config)

    assert first_document.source_id == second_document.source_id
    assert first_chunk.chunk_id != second_chunk.chunk_id


def test_chunking_config_rejects_overlap_greater_than_or_equal_to_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_chars"):
        ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=10)


def test_chunk_document_returns_single_chunk_for_short_document() -> None:
    document = make_source_document("short")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size_chars=100, chunk_overlap_chars=10),
    )

    assert len(chunks) == 1
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)
    assert chunks[0].text == "short"
