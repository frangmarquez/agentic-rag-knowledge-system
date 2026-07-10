"""Qdrant indexing helpers."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from agentic_rag_knowledge_system.embeddings import EmbeddingModel, VectorRecord
from agentic_rag_knowledge_system.ingestion.models import DocumentChunk
from agentic_rag_knowledge_system.retrieval.schemas import RetrievalHit, RetrievalResult
from agentic_rag_knowledge_system.settings import get_settings

QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_BM25_VECTOR_NAME = "bm25"
QDRANT_BM25_MODEL = "Qdrant/bm25"


def qdrant_point_id(vector_id: str) -> str:
    """Return a deterministic Qdrant-compatible point UUID."""

    return str(uuid5(NAMESPACE_URL, vector_id))


def build_qdrant_points(
    records: Sequence[VectorRecord],
    chunks: Sequence[DocumentChunk],
) -> list[object]:
    """Build Qdrant PointStruct objects from vector records and chunks."""

    try:
        from qdrant_client.models import Document, PointStruct
    except ImportError as exc:
        raise RuntimeError("Install qdrant-client to build Qdrant points.") from exc

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_by_id) != len(chunks):
        raise ValueError("chunks must have unique chunk IDs")

    points = []
    for record in records:
        chunk = chunk_by_id.get(record.chunk_id)
        if chunk is None:
            raise ValueError(f"missing chunk for vector record: {record.chunk_id}")

        points.append(
            PointStruct(
                id=qdrant_point_id(record.vector_id),
                vector={
                    QDRANT_DENSE_VECTOR_NAME: list(record.vector),
                    QDRANT_BM25_VECTOR_NAME: Document(
                        text=chunk.text,
                        model=QDRANT_BM25_MODEL,
                    ),
                },
                payload={
                    "vector_id": record.vector_id,
                    "chunk_id": record.chunk_id,
                    "source_id": record.source_id,
                    "source_uri": record.source_uri,
                    "chunk_index": record.chunk_index,
                    "model_name": record.model_name,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "content_hash": chunk.content_hash,
                    "metadata": dict(chunk.metadata),
                },
            )
        )

    return points


def retrieval_hit_from_qdrant_point(point: object) -> RetrievalHit:
    """Convert a Qdrant scored point payload into the local retrieval schema."""

    payload = getattr(point, "payload", None)
    if not isinstance(payload, dict):
        raise ValueError("Qdrant point is missing payload")

    try:
        return RetrievalHit(
            chunk_id=str(_payload_value(payload, "chunk_id")),
            source_id=str(_payload_value(payload, "source_id")),
            source_uri=str(_payload_value(payload, "source_uri")),
            chunk_index=int(_payload_value(payload, "chunk_index")),
            score=float(getattr(point, "score")),
            text=str(_payload_value(payload, "text")),
            metadata=dict(payload.get("metadata") or {}),
        )
    except KeyError as exc:
        raise ValueError(f"Qdrant point payload is missing {exc.args[0]}") from exc


def _payload_value(payload: dict, key: str) -> object:
    value = payload[key]
    if value is None:
        raise ValueError(f"Qdrant point payload is missing {key}")
    return value


class QdrantVectorIndexer:
    """Create a Qdrant collection and upsert chunk vectors."""

    def __init__(
        self,
        *,
        client: object | None = None,
        url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Install qdrant-client to use Qdrant.") from exc
            client = QdrantClient(url=url or settings.qdrant_url)
        self.client = client

    def recreate_collection(self, *, vector_size: int) -> None:
        """Create an empty cosine-distance collection, replacing any existing one."""

        if vector_size < 1:
            raise ValueError("vector_size must be >= 1")

        from qdrant_client.models import (
            Distance,
            Modifier,
            SparseVectorParams,
            VectorParams,
        )

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                QDRANT_DENSE_VECTOR_NAME: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                QDRANT_BM25_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
        chunks: Sequence[DocumentChunk],
    ) -> int:
        """Upsert vector records and chunk payloads into Qdrant."""

        if not records:
            return 0

        points = build_qdrant_points(records, chunks)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)


class QdrantHybridRetriever:
    """Qdrant native dense+BM25 retrieval with RRF fusion."""

    def __init__(
        self,
        *,
        model: EmbeddingModel,
        client: object | None = None,
        url: str | None = None,
        collection_name: str | None = None,
        candidate_k: int = 10,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")

        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        self._model = model
        self._candidate_k = candidate_k
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Install qdrant-client to use Qdrant.") from exc
            client = QdrantClient(url=url or settings.qdrant_url)
        self.client = client

    def search(
        self,
        query: str,
        *,
        query_id: str = "ad_hoc",
        top_k: int = 5,
    ) -> RetrievalResult:
        """Embed a query and return Qdrant-native RRF hybrid matches."""

        return self.search_by_vector(
            self._model.embed_query(query),
            query=query,
            query_id=query_id,
            top_k=top_k,
        )

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        *,
        query: str,
        query_id: str = "ad_hoc",
        top_k: int = 5,
    ) -> RetrievalResult:
        """Return top-k dense+BM25 RRF matches from Qdrant."""

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        vector = tuple(float(value) for value in query_vector)
        if len(vector) != self._model.dimension:
            raise ValueError("query vector dimension must match the embedding model")

        from qdrant_client import models

        candidate_k = max(top_k, self._candidate_k)
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=list(vector),
                    using=QDRANT_DENSE_VECTOR_NAME,
                    limit=candidate_k,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=QDRANT_BM25_MODEL),
                    using=QDRANT_BM25_VECTOR_NAME,
                    limit=candidate_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        hits = tuple(
            retrieval_hit_from_qdrant_point(point)
            for point in getattr(response, "points")
        )
        return RetrievalResult(query_id=query_id, query=query, hits=hits)
