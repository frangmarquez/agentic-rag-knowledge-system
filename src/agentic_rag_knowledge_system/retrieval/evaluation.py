"""Retrieval evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agentic_rag_knowledge_system.ingestion.models import DocumentChunk
from agentic_rag_knowledge_system.ingestion.serialization import read_jsonl
from agentic_rag_knowledge_system.retrieval.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from agentic_rag_knowledge_system.retrieval.schemas import (
    RelevanceQuery,
    RetrievalEvaluation,
    RetrievalResult,
)


class _Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        query_id: str = "ad_hoc",
        top_k: int = 5,
    ) -> RetrievalResult:
        """Return ranked hits for one query."""


def load_relevance_queries(path: str | Path) -> list[RelevanceQuery]:
    """Load seed relevance queries from JSONL."""

    return read_jsonl(path, RelevanceQuery)


def resolve_relevant_chunk_ids(
    query: RelevanceQuery,
    chunks: list[DocumentChunk] | tuple[DocumentChunk, ...],
) -> frozenset[str]:
    """Resolve phrase-level relevance hints to chunk IDs."""

    relevant_terms = tuple(term.casefold() for term in query.relevant_terms)
    return frozenset(
        chunk.chunk_id
        for chunk in chunks
        if any(term in chunk.text.casefold() for term in relevant_terms)
    )


def evaluate_retrieval(
    *,
    retriever: _Retriever,
    queries: list[RelevanceQuery] | tuple[RelevanceQuery, ...],
    chunks: list[DocumentChunk] | tuple[DocumentChunk, ...],
    k: int,
) -> RetrievalEvaluation:
    """Evaluate retrieval with recall@k and MRR."""

    if k < 1:
        raise ValueError("k must be >= 1")

    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    unresolved_query_ids: list[str] = []

    for query in queries:
        relevant_ids = resolve_relevant_chunk_ids(query, chunks)
        if not relevant_ids:
            unresolved_query_ids.append(query.query_id)

        result = retriever.search(query.query, query_id=query.query_id, top_k=k)
        retrieved_ids = tuple(hit.chunk_id for hit in result.hits)
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=k))
        reciprocal_ranks.append(reciprocal_rank(retrieved_ids, relevant_ids))

    return RetrievalEvaluation(
        query_count=len(queries),
        recall_at_k=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        mrr=mean_reciprocal_rank(reciprocal_ranks),
        unresolved_query_ids=tuple(unresolved_query_ids),
    )
