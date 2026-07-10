"""Document ingestion primitives."""

from agentic_rag_knowledge_system.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
)
from agentic_rag_knowledge_system.ingestion.cache import load_or_create_ingestion_cache
from agentic_rag_knowledge_system.ingestion.loaders import (
    SUPPORTED_TEXT_SUFFIXES,
    load_text_directory,
    load_text_file,
)
from agentic_rag_knowledge_system.ingestion.models import (
    DocumentChunk,
    Metadata,
    SourceDocument,
)
from agentic_rag_knowledge_system.ingestion.pipeline import (
    IngestionResult,
    ingest_text_directory,
)
from agentic_rag_knowledge_system.ingestion.serialization import (
    dump_jsonl,
    load_jsonl,
    read_jsonl,
    records_from_jsonl,
    records_to_jsonl,
    write_jsonl,
)

__all__ = [
    "ChunkingConfig",
    "DocumentChunk",
    "IngestionResult",
    "Metadata",
    "SUPPORTED_TEXT_SUFFIXES",
    "SourceDocument",
    "chunk_document",
    "dump_jsonl",
    "ingest_text_directory",
    "load_or_create_ingestion_cache",
    "load_jsonl",
    "load_text_directory",
    "load_text_file",
    "read_jsonl",
    "records_from_jsonl",
    "records_to_jsonl",
    "write_jsonl",
]
