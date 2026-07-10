"""Deterministic plain-text chunking."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from agentic_rag_knowledge_system.ingestion.models import (
    DocumentChunk,
    SourceDocument,
    compute_content_hash,
    stable_chunk_id,
)


class ChunkingConfig(BaseModel):
    """Character-window chunking parameters."""

    chunk_size_chars: int = Field(default=1000, ge=1)
    chunk_overlap_chars: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        return self


def chunk_document(
    document: SourceDocument,
    *,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split a normalized document into deterministic overlapping chunks."""

    chunking_config = config or ChunkingConfig()
    content = document.content
    chunks: list[DocumentChunk] = []
    start_char = 0
    chunk_index = 0

    while start_char < len(content):
        end_char = min(start_char + chunking_config.chunk_size_chars, len(content))
        chunk_text = content[start_char:end_char]

        if chunk_text.strip():
            content_hash = compute_content_hash(chunk_text)
            chunk_id = stable_chunk_id(
                source_id=document.source_id,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                content_hash=content_hash,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=document.source_id,
                    source_uri=document.source_uri,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    content_hash=content_hash,
                    metadata=dict(document.metadata),
                )
            )
            chunk_index += 1

        if end_char == len(content):
            break

        start_char = end_char - chunking_config.chunk_overlap_chars

    return chunks
