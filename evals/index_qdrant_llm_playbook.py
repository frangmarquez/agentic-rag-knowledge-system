"""Index llm-playbook Qwen3 dense and Qdrant BM25 vectors."""

from __future__ import annotations

from pathlib import Path

from agentic_rag_knowledge_system.embeddings import (
    Qwen3EmbeddingModel,
    VectorRecord,
    build_vector_records,
)
from agentic_rag_knowledge_system.ingestion.cache import load_or_create_ingestion_cache
from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.serialization import read_jsonl, write_jsonl
from agentic_rag_knowledge_system.retrieval.qdrant_store import QdrantVectorIndexer
from agentic_rag_knowledge_system.settings import get_settings

ROOT = Path(__file__).parents[1]
CORPUS_DIR = ROOT / "examples" / "corpus"
CACHE_DIR = ROOT / "data" / "processed" / "llm_playbook_ingestion"
VECTOR_CACHE = CACHE_DIR / "qwen3_vectors.jsonl"


def main() -> None:
    ingestion = load_or_create_ingestion_cache(
        CORPUS_DIR,
        CACHE_DIR,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
        metadata={"corpus": "llm-playbook"},
    )
    if VECTOR_CACHE.exists():
        records = tuple(read_jsonl(VECTOR_CACHE, VectorRecord))
    else:
        model = Qwen3EmbeddingModel()
        records = build_vector_records(ingestion.chunks, model=model)
        write_jsonl(records, VECTOR_CACHE)

    settings = get_settings()
    indexer = QdrantVectorIndexer()
    indexer.recreate_collection(vector_size=records[0].dimension)
    indexed_count = indexer.upsert(records, ingestion.chunks)

    print(f"qdrant_url: {settings.qdrant_url}")
    print(f"collection: {indexer.collection_name}")
    print(f"vector_size: {records[0].dimension}")
    print(f"indexed_count: {indexed_count}")


if __name__ == "__main__":
    main()
