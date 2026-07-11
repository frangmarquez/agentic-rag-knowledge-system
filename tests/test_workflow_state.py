import pytest

from agentic_rag_knowledge_system.evidence import EvidenceContext, EvidenceItem
from agentic_rag_knowledge_system.retrieval.schemas import RetrievalHit, RetrievalResult
from agentic_rag_knowledge_system.workflow.state import (
    AgentState,
    GroundingVerdict,
    QueryAnalysis,
    ReformulatedQuery,
)


def _evidence_item(citation_id: str = "E1") -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        rank=1,
        retriever="qdrant_rrf",
        chunk_id="chunk-1",
        source_id="source-1",
        source_uri="source.md",
        chunk_index=0,
        score=0.9,
        quote="Supported evidence.",
    )


def test_agent_state_tracks_workflow_fields() -> None:
    evidence = _evidence_item()
    state = AgentState(
        query_id="q1",
        user_query=" How should I evaluate RAG? ",
        analysis=QueryAnalysis(
            normalized_query="How should I evaluate RAG?",
            needs_retrieval=True,
            reason="Requires corpus evidence.",
        ),
        reformulated_queries=(
            ReformulatedQuery(
                query="RAG evaluation metrics",
                reason="Focus retrieval on evaluation.",
            ),
        ),
        retrieval_result=RetrievalResult(
            query_id="q1",
            query="RAG evaluation metrics",
            hits=(
                RetrievalHit(
                    chunk_id="chunk-1",
                    source_id="source-1",
                    source_uri="source.md",
                    chunk_index=0,
                    score=0.9,
                    text="Supported evidence.",
                ),
            ),
        ),
        evidence=(evidence,),
        evidence_context=EvidenceContext(
            text="[E1] source=source.md\nSupported evidence.",
            evidence=(evidence,),
            max_chars=1_000,
        ),
        answer="Use recall@k and MRR.",
        grounding=GroundingVerdict(status="supported", reason="Answer cites E1."),
    )

    assert state.user_query == "How should I evaluate RAG?"
    assert state.can_retry is True
    assert state.model_dump()["grounding"]["status"] == "supported"


def test_agent_state_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="blank"):
        AgentState(user_query=" ")


def test_agent_state_rejects_retry_count_over_limit() -> None:
    with pytest.raises(ValueError, match="retry_count"):
        AgentState(user_query="query", retry_count=2, max_retries=1)


def test_agent_state_rejects_context_outside_state_evidence() -> None:
    evidence = _evidence_item("E1")
    outside_evidence = _evidence_item("E2")

    with pytest.raises(ValueError, match="state evidence"):
        AgentState(
            user_query="query",
            evidence=(evidence,),
            evidence_context=EvidenceContext(
                text="[E2] source=source.md\nOther evidence.",
                evidence=(outside_evidence,),
                max_chars=1_000,
            ),
        )


def test_grounding_verdict_strips_unsupported_claims() -> None:
    verdict = GroundingVerdict(
        status="unsupported",
        reason="Claim is not in evidence.",
        unsupported_claims=(" unsupported ", " "),
    )

    assert verdict.unsupported_claims == ("unsupported",)
