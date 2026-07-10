"""Simple JSONL ingestion cache."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.models import (
    DocumentChunk,
    MetadataValue,
    SourceDocument,
)
from agentic_rag_knowledge_system.ingestion.pipeline import (
    IngestionResult,
    ingest_text_directory,
)
from agentic_rag_knowledge_system.ingestion.serialization import read_jsonl, write_jsonl


def load_or_create_ingestion_cache(
    corpus_dir: str | Path,
    cache_dir: str | Path,
    *,
    chunking_config: ChunkingConfig | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> IngestionResult:
    """Load cached documents/chunks, or ingest and write them once."""

    cache_path = Path(cache_dir)
    documents_path = cache_path / "documents.jsonl"
    chunks_path = cache_path / "chunks.jsonl"

    if documents_path.exists() and chunks_path.exists():
        return IngestionResult(
            documents=tuple(read_jsonl(documents_path, SourceDocument)),
            chunks=tuple(read_jsonl(chunks_path, DocumentChunk)),
        )

    # ponytail: file-existence cache; delete cache_dir when corpus/chunking changes.
    result = ingest_text_directory(
        corpus_dir,
        chunking_config=chunking_config,
        metadata=metadata,
    )
    write_jsonl(result.documents, documents_path)
    write_jsonl(result.chunks, chunks_path)
    return result
