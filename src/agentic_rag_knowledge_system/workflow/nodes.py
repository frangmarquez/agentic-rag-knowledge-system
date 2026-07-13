"""Deterministic workflow nodes.

These are plain functions first; LangGraph can wrap them later.
"""

from __future__ import annotations

import re
from typing import Literal

from agentic_rag_knowledge_system.evidence import (
    EvidenceItem,
    build_evidence_context,
    build_evidence_items,
)
from agentic_rag_knowledge_system.workflow.state import (
    AgentState,
    GroundingVerdict,
    QueryAnalysis,
    ReformulatedQuery,
)

_CITATION_RE = re.compile(r"\[(E[1-9][0-9]*)\]")
_NO_RETRIEVAL_QUERIES = {"hi", "hello", "hey", "thanks", "thank you"}


def analyze_query(state: AgentState) -> AgentState:
    """Normalize the user query and decide whether corpus retrieval is needed."""

    normalized = _normalize_query(state.user_query)
    simple_query = normalized.lower().rstrip(".!?")
    needs_retrieval = simple_query not in _NO_RETRIEVAL_QUERIES
    reason = (
        "Simple conversational query; corpus retrieval is not needed."
        if not needs_retrieval
        else "Corpus evidence is needed for a grounded answer."
    )
    return state.model_copy(
        update={
            "analysis": QueryAnalysis(
                normalized_query=normalized,
                needs_retrieval=needs_retrieval,
                reason=reason,
            )
        }
    )


def route_after_analysis(state: AgentState) -> Literal["retrieve", "answer"]:
    """Return the next path after query analysis."""

    if state.analysis is None:
        raise ValueError("analysis is required before routing")
    return "retrieve" if state.analysis.needs_retrieval else "answer"


def reformulate_query(state: AgentState) -> AgentState:
    """Create the baseline retrieval query."""

    if state.analysis is not None and not state.analysis.needs_retrieval:
        return state.model_copy(update={"reformulated_queries": ()})

    query = (
        state.analysis.normalized_query
        if state.analysis
        else _normalize_query(state.user_query)
    )
    return state.model_copy(
        update={
            "reformulated_queries": (
                ReformulatedQuery(
                    query=query,
                    reason="Use the normalized user query as the baseline retrieval query.",
                ),
            )
        }
    )


def retrieve_evidence(
    state: AgentState,
    retriever: object,
    *,
    top_k: int = 5,
    max_context_chars: int = 4_000,
    retriever_name: str = "qdrant_rrf",
) -> AgentState:
    """Run retrieval and package hits as citation-ready evidence."""

    if state.analysis is not None and not state.analysis.needs_retrieval:
        return state

    query = _retrieval_query(state)
    result = retriever.search(query, query_id=state.query_id, top_k=top_k)
    evidence = build_evidence_items(result, retriever=retriever_name)
    return state.model_copy(
        update={
            "retrieval_result": result,
            "evidence": evidence,
            "evidence_context": build_evidence_context(
                evidence,
                max_chars=max_context_chars,
            ),
        }
    )


def generate_answer(state: AgentState) -> AgentState:
    """Create a deterministic placeholder answer from available evidence."""

    evidence = _available_evidence(state)
    if state.analysis is not None and not state.analysis.needs_retrieval:
        answer = "This query does not require corpus retrieval."
    elif not evidence:
        answer = "I do not have enough corpus evidence to answer this."
    else:
        citations = " ".join(f"[{item.citation_id}]" for item in evidence)
        answer = f"I found relevant corpus evidence for this query in {citations}."

    return state.model_copy(update={"answer": answer})


def check_grounding(state: AgentState) -> AgentState:
    """Check whether the answer cites evidence available in state."""

    if state.answer is None:
        verdict = GroundingVerdict(
            status="unsupported",
            reason="No answer was generated.",
        )
    elif not _available_evidence(state):
        verdict = GroundingVerdict(
            status="insufficient_evidence",
            reason="No evidence is available to support the answer.",
        )
    else:
        verdict = _citation_verdict(state)

    return state.model_copy(update={"grounding": verdict})


def route_after_grounding(state: AgentState) -> Literal["retry", "done"]:
    """Return whether weak grounding should trigger a retry."""

    if state.grounding is None:
        raise ValueError("grounding is required before routing")
    return "retry" if state.grounding.status != "supported" and state.can_retry else "done"


def prepare_retry(state: AgentState) -> AgentState:
    """Clear downstream outputs and increment the retry counter."""

    if (
        state.grounding is None
        or state.grounding.status == "supported"
        or not state.can_retry
    ):
        return state

    return state.model_copy(
        update={
            "retry_count": state.retry_count + 1,
            "retrieval_result": None,
            "evidence": (),
            "evidence_context": None,
            "answer": None,
            "grounding": None,
        }
    )


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _retrieval_query(state: AgentState) -> str:
    if state.reformulated_queries:
        return state.reformulated_queries[-1].query
    if state.analysis is not None:
        return state.analysis.normalized_query
    return _normalize_query(state.user_query)


def _available_evidence(state: AgentState) -> tuple[EvidenceItem, ...]:
    if state.evidence_context is not None:
        return state.evidence_context.evidence
    return state.evidence


def _citation_verdict(state: AgentState) -> GroundingVerdict:
    known_ids = {item.citation_id for item in _available_evidence(state)}
    cited_ids = set(_CITATION_RE.findall(state.answer or ""))
    unknown_ids = sorted(cited_ids - known_ids)

    if not cited_ids:
        return GroundingVerdict(
            status="unsupported",
            reason="The answer does not cite evidence.",
            unsupported_claims=("missing evidence citations",),
        )
    if unknown_ids:
        return GroundingVerdict(
            status="unsupported",
            reason="The answer cites evidence that is not available.",
            unsupported_claims=tuple(
                f"unknown citation {citation_id}" for citation_id in unknown_ids
            ),
        )
    return GroundingVerdict(
        status="supported",
        reason="The answer cites available evidence.",
    )
