import pytest

from agentic_rag_knowledge_system.retrieval.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k() -> None:
    assert recall_at_k(("a", "b", "c"), {"b", "d"}, k=2) == 0.5
    assert recall_at_k(("a", "b", "c"), {"b", "d"}, k=3) == 0.5


def test_recall_at_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(("a",), {"a"}, k=0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(("a", "b", "c"), {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(("a", "b", "c"), {"x"}) == 0.0


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank((1.0, 0.5, 0.0)) == pytest.approx(0.5)
    assert mean_reciprocal_rank(()) == 0.0
