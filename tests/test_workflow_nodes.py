from types import SimpleNamespace

from agentic_rag_knowledge_system.retrieval.schemas import RetrievalHit, RetrievalResult
from agentic_rag_knowledge_system.workflow.nodes import (
    analyze_query,
    check_grounding,
    generate_answer,
    prepare_retry,
    reformulate_query,
    retrieve_evidence,
    route_after_analysis,
    route_after_grounding,
)
from agentic_rag_knowledge_system.workflow.state import AgentState


def _hit(chunk_id: str = "chunk-1") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_id="source-1",
        source_uri="source.md",
        chunk_index=0,
        score=0.9,
        text="Evaluation should measure recall@k and MRR.",
    )


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[SimpleNamespace] = []

    def search(self, query: str, *, query_id: str, top_k: int) -> RetrievalResult:
        self.calls.append(SimpleNamespace(query=query, query_id=query_id, top_k=top_k))
        return self.result


def test_analyze_query_routes_simple_greetings_to_answer() -> None:
    state = analyze_query(AgentState(user_query=" Hello! "))

    assert state.analysis is not None
    assert state.analysis.normalized_query == "Hello!"
    assert state.analysis.needs_retrieval is False
    assert route_after_analysis(state) == "answer"


def test_reformulate_query_uses_normalized_query() -> None:
    state = reformulate_query(
        analyze_query(AgentState(user_query="  How evaluate RAG?  "))
    )

    assert state.reformulated_queries[0].query == "How evaluate RAG?"
    assert route_after_analysis(state) == "retrieve"


def test_retrieve_evidence_builds_result_evidence_and_context() -> None:
    result = RetrievalResult(query_id="q1", query="RAG metrics", hits=(_hit(),))
    retriever = FakeRetriever(result)
    state = reformulate_query(
        analyze_query(AgentState(query_id="q1", user_query="RAG metrics"))
    )

    state = retrieve_evidence(
        state,
        retriever,
        top_k=3,
        max_context_chars=1_000,
    )

    assert retriever.calls[0].query == "RAG metrics"
    assert retriever.calls[0].query_id == "q1"
    assert retriever.calls[0].top_k == 3
    assert state.retrieval_result == result
    assert state.evidence[0].citation_id == "E1"
    assert state.evidence_context is not None
    assert "[E1] source=source.md" in state.evidence_context.text


def test_generate_answer_and_grounding_support_known_citations() -> None:
    result = RetrievalResult(query_id="q1", query="RAG metrics", hits=(_hit(),))
    state = retrieve_evidence(
        reformulate_query(
            analyze_query(AgentState(query_id="q1", user_query="RAG metrics"))
        ),
        FakeRetriever(result),
    )

    state = check_grounding(generate_answer(state))

    assert state.answer is not None
    assert "[E1]" in state.answer
    assert state.grounding is not None
    assert state.grounding.status == "supported"
    assert route_after_grounding(state) == "done"


def test_grounding_retry_path_clears_downstream_outputs() -> None:
    state = AgentState(user_query="RAG metrics", answer="Unsupported answer.")
    state = check_grounding(state)

    assert state.grounding is not None
    assert state.grounding.status == "insufficient_evidence"
    assert route_after_grounding(state) == "retry"

    retry_state = prepare_retry(state)

    assert retry_state.retry_count == 1
    assert retry_state.retrieval_result is None
    assert retry_state.evidence == ()
    assert retry_state.evidence_context is None
    assert retry_state.answer is None
    assert retry_state.grounding is None


def test_grounding_rejects_unknown_citations() -> None:
    result = RetrievalResult(query_id="q1", query="RAG metrics", hits=(_hit(),))
    state = retrieve_evidence(
        reformulate_query(
            analyze_query(AgentState(query_id="q1", user_query="RAG metrics"))
        ),
        FakeRetriever(result),
    ).model_copy(update={"answer": "Use recall@k [E2]."})

    state = check_grounding(state)

    assert state.grounding is not None
    assert state.grounding.status == "unsupported"
    assert state.grounding.unsupported_claims == ("unknown citation E2",)
