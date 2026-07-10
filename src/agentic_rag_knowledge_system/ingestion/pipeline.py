"""Ingestion pipeline boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

from agentic_rag_knowledge_system.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
)
from agentic_rag_knowledge_system.ingestion.loaders import load_text_directory
from agentic_rag_knowledge_system.ingestion.models import (
    DocumentChunk,
    MetadataValue,
    SourceDocument,
)


class IngestionResult(BaseModel):
    """Documents and chunks produced by one ingestion run."""

    model_config = ConfigDict(frozen=True)

    documents: tuple[SourceDocument, ...]
    chunks: tuple[DocumentChunk, ...]

    @computed_field
    @property
    def document_count(self) -> int:
        return len(self.documents)

    @computed_field
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def ingest_text_directory(
    directory: str | Path,
    *,
    chunking_config: ChunkingConfig | None = None,
    recursive: bool = True,
    metadata: Mapping[str, MetadataValue] | None = None,
    encoding: str = "utf-8",
) -> IngestionResult:
    """Load and chunk all supported text documents from a directory."""

    documents = tuple(
        load_text_directory(
            directory,
            recursive=recursive,
            metadata=metadata,
            encoding=encoding,
        )
    )
    chunks = tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, config=chunking_config)
    )

    return IngestionResult(documents=documents, chunks=chunks)
