from app.clients.embeddings import EmbeddingClient
from app.clients.qdrant import QdrantRepository

from app.core.models import DocumentChunk


class IngestionService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        qdrant_repository: QdrantRepository,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_repository = qdrant_repository

    def ingest(self, chunk: DocumentChunk) -> None:
        vector = self.embedding_client.embed(chunk.text)

        self.qdrant_repository.upsert(
            chunk=chunk,
            vector=vector,
        )
