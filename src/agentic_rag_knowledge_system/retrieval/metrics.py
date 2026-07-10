"""Deterministic retrieval metrics."""

from __future__ import annotations


def recall_at_k(
    retrieved_ids: list[str] | tuple[str, ...],
    relevant_ids: set[str] | frozenset[str],
    *,
    k: int,
) -> float:
    """Compute recall@k for one query."""

    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & set(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str] | tuple[str, ...],
    relevant_ids: set[str] | frozenset[str],
) -> float:
    """Compute reciprocal rank for one query."""

    if not relevant_ids:
        return 0.0

    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(scores: list[float] | tuple[float, ...]) -> float:
    """Compute mean reciprocal rank from per-query reciprocal ranks."""

    if not scores:
        return 0.0

    return sum(scores) / len(scores)
