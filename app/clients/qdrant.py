import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.core.models import DocumentChunk, SearchResult


class QdrantRepository:
    def __init__(
            self,
            collection_name: str | None = None,
    ) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url
        )

        self.collection_name = (
            collection_name
            if collection_name is not None
            else settings.qdrant_collection
        )

    def ensure_collection(self) -> None:
        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_size,
                distance=Distance.COSINE,
            ),
        )

    def collection_exists(self) -> bool:
        collections = self.client.get_collections()

        return self.collection_name in {
            collection.name
            for collection in collections.collections
        }

    def upsert(
        self,
        chunk: DocumentChunk,
        vector: list[float],
    ) -> None:
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk.id,
            )
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                    },
                )
            ],
        )

    def count(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def _payload_to_chunk(
        self,
        payload: dict,
    ) -> DocumentChunk:
        return DocumentChunk(
            id=payload["chunk_id"],
            document_id=payload["document_id"],
            chunk_index=payload["chunk_index"],
            text=payload["text"],
            source=payload["source"],
            metadata=payload["metadata"],
        )

    def search(
            self,
            vector: list[float],
            limit: int = 5,
    ) -> list[SearchResult]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            with_payload=True,
            limit=limit,
        ).points

        return [
            SearchResult(
                score=float(result.score),
                document=self._payload_to_chunk(result.payload),
            )
            for result in results
        ]

    def get_all_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
        )

        for point in points:
            chunks.append(
                self._payload_to_chunk(point.payload)
            )

        return chunks
