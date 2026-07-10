from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag_knowledge_system.embeddings import FakeEmbeddingModel, build_vector_records
from agentic_rag_knowledge_system.ingestion.chunking import ChunkingConfig
from agentic_rag_knowledge_system.ingestion.pipeline import ingest_text_directory
from agentic_rag_knowledge_system.retrieval.qdrant_store import (
    QDRANT_BM25_MODEL,
    QDRANT_BM25_VECTOR_NAME,
    QDRANT_DENSE_VECTOR_NAME,
    QdrantHybridRetriever,
    QdrantVectorIndexer,
    build_qdrant_points,
    qdrant_point_id,
    retrieval_hit_from_qdrant_point,
)

EXAMPLE_CORPUS = Path(__file__).parents[1] / "examples" / "corpus"


class FakeQdrantClient:
    def __init__(self, *, exists: bool = False) -> None:
        self.exists = exists
        self.deleted: list[str] = []
        self.created: list[object] = []
        self.upserted: list[object] = []
        self.queried: list[object] = []
        self.query_points_response = SimpleNamespace(points=())

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)

    def create_collection(self, **kwargs: object) -> None:
        self.created.append(kwargs)

    def upsert(self, **kwargs: object) -> None:
        self.upserted.append(kwargs)

    def query_points(self, **kwargs: object) -> object:
        self.queried.append(kwargs)
        return self.query_points_response


def test_qdrant_point_id_is_stable_uuid() -> None:
    point_id = qdrant_point_id("vec_test")

    assert point_id == qdrant_point_id("vec_test")
    assert len(point_id) == 36


def test_build_qdrant_points_preserves_chunk_payload() -> None:
    ingestion = ingest_text_directory(
        EXAMPLE_CORPUS,
        chunking_config=ChunkingConfig(chunk_size_chars=2500, chunk_overlap_chars=250),
        metadata={"corpus": "llm-playbook"},
    )
    model = FakeEmbeddingModel(dimension=4)
    records = build_vector_records(ingestion.chunks[:1], model=model)

    point = build_qdrant_points(records, ingestion.chunks)[0]

    assert point.id == qdrant_point_id(records[0].vector_id)
    assert point.vector[QDRANT_DENSE_VECTOR_NAME] == list(records[0].vector)
    assert point.vector[QDRANT_BM25_VECTOR_NAME].text == ingestion.chunks[0].text
    assert point.vector[QDRANT_BM25_VECTOR_NAME].model == QDRANT_BM25_MODEL
    assert point.payload["vector_id"] == records[0].vector_id
    assert point.payload["chunk_id"] == records[0].chunk_id
    assert point.payload["source_uri"] == "llm-playbook.md"
    assert point.payload["text"] == ingestion.chunks[0].text
    assert point.payload["metadata"]["corpus"] == "llm-playbook"


def test_build_qdrant_points_rejects_missing_chunks() -> None:
    ingestion = ingest_text_directory(EXAMPLE_CORPUS)
    model = FakeEmbeddingModel(dimension=4)
    records = build_vector_records(ingestion.chunks[:1], model=model)
    broken = records[0].model_copy(update={"chunk_id": "missing"})

    with pytest.raises(ValueError, match="missing chunk"):
        build_qdrant_points((broken,), ingestion.chunks)


def test_qdrant_indexer_recreates_collection_and_upserts_points() -> None:
    ingestion = ingest_text_directory(EXAMPLE_CORPUS)
    model = FakeEmbeddingModel(dimension=4)
    records = build_vector_records(ingestion.chunks[:2], model=model)
    client = FakeQdrantClient(exists=True)
    indexer = QdrantVectorIndexer(client=client, collection_name="test_collection")

    indexer.recreate_collection(vector_size=4)
    count = indexer.upsert(records, ingestion.chunks)

    assert client.deleted == ["test_collection"]
    assert client.created[0]["collection_name"] == "test_collection"
    assert QDRANT_DENSE_VECTOR_NAME in client.created[0]["vectors_config"]
    assert QDRANT_BM25_VECTOR_NAME in client.created[0]["sparse_vectors_config"]
    assert count == 2
    assert client.upserted[0]["collection_name"] == "test_collection"
    assert len(client.upserted[0]["points"]) == 2


def test_retrieval_hit_from_qdrant_point_preserves_payload() -> None:
    ingestion = ingest_text_directory(EXAMPLE_CORPUS)
    model = FakeEmbeddingModel(dimension=4)
    records = build_vector_records(ingestion.chunks[:1], model=model)
    payload = build_qdrant_points(records, ingestion.chunks)[0].payload

    hit = retrieval_hit_from_qdrant_point(SimpleNamespace(score=0.75, payload=payload))

    assert hit.chunk_id == records[0].chunk_id
    assert hit.source_uri == "llm-playbook.md"
    assert hit.score == pytest.approx(0.75)
    assert hit.text == ingestion.chunks[0].text


def test_retrieval_hit_from_qdrant_point_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="missing chunk_id"):
        retrieval_hit_from_qdrant_point(SimpleNamespace(score=1.0, payload={}))


def test_qdrant_hybrid_retriever_uses_native_rrf_prefetches() -> None:
    ingestion = ingest_text_directory(EXAMPLE_CORPUS)
    model = FakeEmbeddingModel(dimension=4)
    records = build_vector_records(ingestion.chunks[:1], model=model)
    payload = build_qdrant_points(records, ingestion.chunks)[0].payload
    client = FakeQdrantClient()
    client.query_points_response = SimpleNamespace(
        points=(SimpleNamespace(score=0.8, payload=payload),)
    )
    retriever = QdrantHybridRetriever(
        model=model,
        client=client,
        collection_name="test_collection",
        candidate_k=7,
    )

    result = retriever.search("keyword query", query_id="q1", top_k=3)

    query_call = client.queried[0]
    assert query_call["collection_name"] == "test_collection"
    assert query_call["limit"] == 3
    assert query_call["with_payload"] is True
    assert [prefetch.using for prefetch in query_call["prefetch"]] == [
        QDRANT_DENSE_VECTOR_NAME,
        QDRANT_BM25_VECTOR_NAME,
    ]
    assert query_call["prefetch"][0].limit == 7
    assert query_call["prefetch"][1].query.text == "keyword query"
    assert query_call["prefetch"][1].query.model == QDRANT_BM25_MODEL
    assert query_call["query"].fusion.value == "rrf"
    assert result.hits[0].chunk_id == records[0].chunk_id


def test_qdrant_hybrid_retriever_validates_query_vector() -> None:
    client = FakeQdrantClient()
    model = FakeEmbeddingModel(dimension=4)
    retriever = QdrantHybridRetriever(
        model=model,
        client=client,
        collection_name="test_collection",
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.search_by_vector((1.0, 0.0, 0.0, 0.0), query="x", top_k=0)

    with pytest.raises(ValueError, match="dimension"):
        retriever.search_by_vector((1.0, 0.0), query="x", top_k=1)

    with pytest.raises(ValueError, match="candidate_k"):
        QdrantHybridRetriever(
            model=model,
            client=client,
            collection_name="test_collection",
            candidate_k=0,
        )
