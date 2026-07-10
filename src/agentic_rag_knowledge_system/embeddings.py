"""Embedding models and vector record helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from agentic_rag_knowledge_system.ingestion.models import DocumentChunk, Metadata
from agentic_rag_knowledge_system.settings import get_settings

QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_QUERY_TASK = (
    "Given a user query, retrieve relevant passages from the AI engineering "
    "handbook that answer the query."
)


class EmbeddingModel(Protocol):
    model_name: str
    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed document texts."""

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed a retrieval query."""


class VectorRecord(BaseModel):
    """Vector payload for one chunk."""

    model_config = ConfigDict(frozen=True)

    vector_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    model_name: str = Field(min_length=1)
    vector: tuple[float, ...] = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @computed_field
    @property
    def dimension(self) -> int:
        return len(self.vector)


def stable_vector_id(*, chunk_id: str, model_name: str) -> str:
    digest = hashlib.sha256(f"{model_name}:{chunk_id}".encode("utf-8")).hexdigest()
    return f"vec_{digest[:20]}"


def format_qwen3_query(query: str, *, task: str = QWEN3_QUERY_TASK) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    return f"Instruct: {task}\nQuery: {cleaned}"


def build_vector_records(
    chunks: Sequence[DocumentChunk],
    *,
    model: EmbeddingModel,
) -> tuple[VectorRecord, ...]:
    vectors = model.embed_texts(tuple(chunk.text for chunk in chunks))
    if len(vectors) != len(chunks):
        raise ValueError("embedding count must match chunk count")

    records: list[VectorRecord] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != model.dimension:
            raise ValueError("embedding dimension must match model dimension")

        records.append(
            VectorRecord(
                vector_id=stable_vector_id(
                    chunk_id=chunk.chunk_id,
                    model_name=model.model_name,
                ),
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_uri=chunk.source_uri,
                chunk_index=chunk.chunk_index,
                model_name=model.model_name,
                vector=vector,
                metadata=dict(chunk.metadata),
            )
        )

    return tuple(records)


class FakeEmbeddingModel:
    """Deterministic embedding model for fast tests."""

    def __init__(self, *, dimension: int = 8, model_name: str = "fake-embedding") -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = dimension
        self.model_name = model_name

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, query: str) -> tuple[float, ...]:
        return self._embed(format_qwen3_query(query))

    def _embed(self, text: str) -> tuple[float, ...]:
        values: list[float] = []
        seed = text.encode("utf-8")
        while len(values) < self.dimension:
            seed = hashlib.sha256(seed).digest()
            values.extend((byte / 127.5) - 1.0 for byte in seed)

        vector = values[: self.dimension]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)


class Qwen3EmbeddingModel:
    """Qwen3 embedding adapter backed by sentence-transformers."""

    def __init__(
        self,
        *,
        model_name: str = QWEN3_EMBEDDING_MODEL,
        dimension: int = 1024,
        query_task: str = QWEN3_QUERY_TASK,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers, transformers, and torch to use "
                "Qwen3EmbeddingModel."
            ) from exc

        settings = get_settings()
        token = hf_token
        if token is None and settings.hf_token is not None:
            token = settings.hf_token.get_secret_value()

        kwargs = {"device": device} if device else {}
        if token:
            kwargs["token"] = token

        self._model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name
        self.dimension = dimension
        self.query_task = query_task

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def embed_query(self, query: str) -> tuple[float, ...]:
        return self._encode((format_qwen3_query(query, task=self.query_task),))[0]

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            truncate_dim=self.dimension,
        )
        rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        return tuple(tuple(float(value) for value in row) for row in rows)
