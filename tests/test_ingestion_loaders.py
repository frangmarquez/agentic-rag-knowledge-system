from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag_knowledge_system.ingestion.loaders import (
    SUPPORTED_TEXT_SUFFIXES,
    load_text_directory,
    load_text_file,
)
from agentic_rag_knowledge_system.ingestion.models import (
    SourceDocument,
    compute_content_hash,
    stable_source_id,
)

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
INVALID_FIXTURES = Path(__file__).parent / "fixtures" / "invalid"


def test_load_text_file_builds_normalized_source_document() -> None:
    document_path = FIXTURE_CORPUS / "guide.md"

    document = load_text_file(
        document_path,
        corpus_root=FIXTURE_CORPUS,
        metadata={"domain": "docs"},
    )

    assert document.source_uri == "guide.md"
    assert document.source_id == stable_source_id("guide.md")
    assert document.title == "Agentic RAG"
    assert document.content == "# Agentic RAG\n\nThis is a test document.\n"
    assert document.content_hash == compute_content_hash(document.content)
    assert document.metadata == {
        "file_name": "guide.md",
        "file_extension": ".md",
        "domain": "docs",
    }


def test_load_text_directory_returns_supported_files_in_stable_order() -> None:
    documents = load_text_directory(FIXTURE_CORPUS)

    assert [document.source_uri for document in documents] == [
        "a.md",
        "b.txt",
        "guide.md",
        "nested/c.markdown",
    ]
    assert {document.metadata["file_extension"] for document in documents}.issubset(
        SUPPORTED_TEXT_SUFFIXES
    )


def test_load_text_file_rejects_empty_documents() -> None:
    document_path = INVALID_FIXTURES / "empty.txt"

    with pytest.raises(ValueError, match="empty document"):
        load_text_file(document_path, corpus_root=INVALID_FIXTURES)


def test_load_text_file_rejects_unsupported_suffix() -> None:
    document_path = FIXTURE_CORPUS / "notes.pdf"

    with pytest.raises(ValueError, match="Unsupported text document suffix"):
        load_text_file(document_path, corpus_root=FIXTURE_CORPUS)


def test_source_document_validates_content_hash() -> None:
    with pytest.raises(ValidationError, match="content_hash must match"):
        SourceDocument(
            source_id="src_test",
            source_uri="test.md",
            title="Test",
            content="real content",
            content_hash=compute_content_hash("different content"),
        )
