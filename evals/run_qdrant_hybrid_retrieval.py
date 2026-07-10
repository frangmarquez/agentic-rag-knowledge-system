"""Run Qdrant-native dense+BM25 RRF retrieval evaluation."""

from __future__ import annotations

from pathlib import Path

from agentic_rag_knowledge_system.embeddings import Qwen3EmbeddingModel
from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.pipeline import ingest_text_directory
from agentic_rag_knowledge_system.retrieval.evaluation import (
    evaluate_retrieval,
    load_relevance_queries,
)
from agentic_rag_knowledge_system.retrieval.qdrant_store import QdrantHybridRetriever

ROOT = Path(__file__).parents[1]
CORPUS_DIR = ROOT / "examples" / "corpus"
RELEVANCE_FILE = ROOT / "evals" / "relevance" / "llm_playbook_seed.jsonl"


def main() -> None:
    ingestion = ingest_text_directory(
        CORPUS_DIR,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
    )
    retriever = QdrantHybridRetriever(model=Qwen3EmbeddingModel())
    queries = load_relevance_queries(RELEVANCE_FILE)
    evaluation = evaluate_retrieval(
        retriever=retriever,
        queries=queries,
        chunks=ingestion.chunks,
        k=5,
    )

    print(f"query_count: {evaluation.query_count}")
    print(f"recall@5: {evaluation.recall_at_k:.3f}")
    print(f"mrr: {evaluation.mrr:.3f}")
    print(f"unresolved_query_ids: {list(evaluation.unresolved_query_ids)}")


if __name__ == "__main__":
    main()
