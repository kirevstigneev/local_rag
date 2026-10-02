from app.core.config import settings
from app.clients.embeddings import EmbeddingClient
from app.core.models import SearchResult
from app.clients.qdrant import QdrantRepository


class Retriever:
    def __init__(
            self,
            embedding_client: EmbeddingClient,
            qdrant_repository: QdrantRepository,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_repository = qdrant_repository

    def search(
            self,
            query: str,
            limit: int = 5
    ) -> list[SearchResult]:
        vector = self.embedding_client.embed(query)

        results = self.qdrant_repository.search(
            vector=vector,
            limit=limit
        )

        return [
            result
            for result in results
            if result.score >= settings.retrieval_min_score
        ]
