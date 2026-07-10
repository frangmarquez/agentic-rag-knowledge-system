"""Typed retrieval schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag_knowledge_system.ingestion.models import DocumentChunk, Metadata


class RelevanceQuery(BaseModel):
    """A seed relevance query with phrase-level relevance hints."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_terms: tuple[str, ...] = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be whitespace-only")
        return value

    @field_validator("relevant_terms")
    @classmethod
    def validate_relevant_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values if value.strip())
        if not normalized:
            raise ValueError("relevant_terms must contain at least one non-empty term")
        return normalized


class RetrievalHit(BaseModel):
    """One scored chunk returned by a retriever."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    score: float
    text: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: DocumentChunk, *, score: float) -> "RetrievalHit":
        return cls(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_uri=chunk.source_uri,
            chunk_index=chunk.chunk_index,
            score=score,
            text=chunk.text,
            metadata=dict(chunk.metadata),
        )


class RetrievalResult(BaseModel):
    """Retriever output for one query."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    hits: tuple[RetrievalHit, ...] = Field(default_factory=tuple)


class RetrievalEvaluation(BaseModel):
    """Aggregate retrieval metrics for one evaluation run."""

    model_config = ConfigDict(frozen=True)

    query_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    unresolved_query_ids: tuple[str, ...] = Field(default_factory=tuple)
