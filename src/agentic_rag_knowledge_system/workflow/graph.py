"""Minimal LangGraph workflow wiring."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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


def build_workflow(
    retriever: object,
    *,
    top_k: int = 5,
    max_context_chars: int = 4_000,
    retriever_name: str = "qdrant_rrf",
) -> CompiledStateGraph:
    """Compile the first bounded Agentic RAG workflow."""

    def retrieve(state: AgentState) -> AgentState:
        return retrieve_evidence(
            state,
            retriever,
            top_k=top_k,
            max_context_chars=max_context_chars,
            retriever_name=retriever_name,
        )

    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("retrieve_evidence", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("prepare_retry", prepare_retry)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {"retrieve": "reformulate_query", "answer": "generate_answer"},
    )
    graph.add_edge("reformulate_query", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        _route_after_answer,
        {"check": "check_grounding", "done": END},
    )
    graph.add_conditional_edges(
        "check_grounding",
        route_after_grounding,
        {"retry": "prepare_retry", "done": END},
    )
    graph.add_edge("prepare_retry", "reformulate_query")
    return graph.compile()


def run_workflow(workflow: CompiledStateGraph, state: AgentState) -> AgentState:
    """Invoke a compiled workflow and return validated project state."""

    return AgentState.model_validate(workflow.invoke(state))


def _route_after_answer(state: AgentState) -> Literal["check", "done"]:
    if state.analysis is not None and not state.analysis.needs_retrieval:
        return "done"
    return "check"


class _DevRetriever:
    def search(self, query: str, *, query_id: str, top_k: int) -> RetrievalResult:
        return RetrievalResult(
            query_id=query_id,
            query=query,
            hits=(
                RetrievalHit(
                    chunk_id="dev-chunk-1",
                    source_id="dev-source",
                    source_uri="langgraph-dev-stub",
                    chunk_index=0,
                    score=1.0,
                    text="This is stub evidence for LangGraph dev visualization.",
                ),
            )[:top_k],
        )


graph = build_workflow(_DevRetriever(), retriever_name="dev_stub")
