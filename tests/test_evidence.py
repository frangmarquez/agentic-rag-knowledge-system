from pathlib import Path

import pytest

from agentic_rag_knowledge_system.evidence import (
    EvidenceItem,
    build_evidence_context,
    build_evidence_items,
)
from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.pipeline import ingest_text_directory
from agentic_rag_knowledge_system.retrieval.schemas import RetrievalHit, RetrievalResult

EXAMPLE_CORPUS = Path(__file__).parents[1] / "examples" / "corpus"


def test_evidence_item_preserves_hit_lineage() -> None:
    chunk = ingest_text_directory(
        EXAMPLE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
        metadata={"corpus": "llm-playbook"},
    ).chunks[0]
    hit = RetrievalHit.from_chunk(chunk, score=0.42)

    evidence = EvidenceItem.from_hit(hit, rank=1, retriever="bm25")

    assert evidence.citation_id == "E1"
    assert evidence.rank == 1
    assert evidence.retriever == "bm25"
    assert evidence.chunk_id == hit.chunk_id
    assert evidence.source_uri == "llm-playbook.md"
    assert evidence.quote == hit.text
    assert evidence.score == 0.42
    assert evidence.metadata["corpus"] == "llm-playbook"


def test_build_evidence_items_preserves_retrieval_order() -> None:
    ingestion = ingest_text_directory(
        EXAMPLE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
    )
    result = RetrievalResult(
        query_id="q",
        query="LangGraph complex stateful agents",
        hits=tuple(
            RetrievalHit.from_chunk(chunk, score=1.0)
            for chunk in ingestion.chunks[:2]
        ),
    )

    evidence = build_evidence_items(result, retriever="bm25")

    assert [item.citation_id for item in evidence] == ["E1", "E2"]
    assert [item.chunk_id for item in evidence] == [
        hit.chunk_id for hit in result.hits
    ]


def test_evidence_rejects_blank_retriever() -> None:
    chunk = ingest_text_directory(EXAMPLE_CORPUS).chunks[0]
    hit = RetrievalHit.from_chunk(chunk, score=1.0)

    with pytest.raises(ValueError, match="value must not be blank"):
        EvidenceItem.from_hit(hit, rank=1, retriever=" ")


def test_build_evidence_context_renders_citation_blocks() -> None:
    ingestion = ingest_text_directory(
        EXAMPLE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
    )
    result = RetrievalResult(
        query_id="q",
        query="LangGraph complex stateful agents",
        hits=tuple(
            RetrievalHit.from_chunk(chunk, score=1.0)
            for chunk in ingestion.chunks[:2]
        ),
    )
    evidence = build_evidence_items(result, retriever="bm25")

    context = build_evidence_context(evidence, max_chars=10_000)

    assert context.evidence == evidence
    assert context.omitted_citation_ids == ()
    assert len(context.text) <= context.max_chars
    assert "[E1] source=llm-playbook.md" in context.text
    assert "[E2] source=llm-playbook.md" in context.text
    assert f"chunk={evidence[0].chunk_index}" in context.text


def test_build_evidence_context_truncates_and_tracks_omissions() -> None:
    first = EvidenceItem(
        citation_id="E1",
        rank=1,
        retriever="hybrid",
        chunk_id="chunk-1",
        source_id="source-1",
        source_uri="source.md",
        chunk_index=0,
        score=1.0,
        quote="a" * 200,
    )
    second = first.model_copy(
        update={"citation_id": "E2", "rank": 2, "chunk_id": "chunk-2"},
    )

    context = build_evidence_context((first, second), max_chars=80)

    assert len(context.text) <= 80
    assert context.evidence == (first,)
    assert context.omitted_citation_ids == ("E2",)
    assert context.text.endswith("...")


def test_build_evidence_context_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        build_evidence_context((), max_chars=0)
