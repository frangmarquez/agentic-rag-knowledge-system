from types import SimpleNamespace

from agentic_rag_knowledge_system.retrieval.schemas import RetrievalHit, RetrievalResult
from agentic_rag_knowledge_system.workflow.graph import build_workflow, run_workflow
from agentic_rag_knowledge_system.workflow.state import AgentState


class FakeRetriever:
    def __init__(self, hits: tuple[RetrievalHit, ...]) -> None:
        self.hits = hits
        self.calls: list[SimpleNamespace] = []

    def search(self, query: str, *, query_id: str, top_k: int) -> RetrievalResult:
        self.calls.append(SimpleNamespace(query=query, query_id=query_id, top_k=top_k))
        return RetrievalResult(query_id=query_id, query=query, hits=self.hits)


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        source_id="source-1",
        source_uri="source.md",
        chunk_index=0,
        score=0.9,
        text="Evaluation should measure recall@k and MRR.",
    )


def test_workflow_runs_retrieval_happy_path() -> None:
    retriever = FakeRetriever((_hit(),))
    workflow = build_workflow(retriever, top_k=3)

    state = run_workflow(workflow, AgentState(query_id="q1", user_query="RAG metrics"))

    assert len(retriever.calls) == 1
    assert retriever.calls[0].query == "RAG metrics"
    assert retriever.calls[0].top_k == 3
    assert state.analysis is not None
    assert state.analysis.needs_retrieval is True
    assert state.evidence[0].citation_id == "E1"
    assert state.answer is not None
    assert "[E1]" in state.answer
    assert state.grounding is not None
    assert state.grounding.status == "supported"


def test_workflow_retries_once_when_evidence_is_missing() -> None:
    retriever = FakeRetriever(())
    workflow = build_workflow(retriever)

    state = run_workflow(
        workflow,
        AgentState(query_id="q1", user_query="RAG metrics", max_retries=1),
    )

    assert len(retriever.calls) == 2
    assert state.retry_count == 1
    assert state.evidence == ()
    assert state.grounding is not None
    assert state.grounding.status == "insufficient_evidence"


def test_workflow_skips_retrieval_for_simple_query() -> None:
    retriever = FakeRetriever((_hit(),))
    workflow = build_workflow(retriever)

    state = run_workflow(workflow, AgentState(user_query="hello"))

    assert retriever.calls == []
    assert state.analysis is not None
    assert state.analysis.needs_retrieval is False
    assert state.answer == "This query does not require corpus retrieval."
    assert state.grounding is None
