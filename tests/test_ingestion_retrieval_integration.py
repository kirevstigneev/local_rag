from app.core.config import settings
from app.ingestion.ingestion import IngestionService
from app.core.models import DocumentChunk
from app.clients.qdrant import QdrantRepository
from app.retrieval.retrieval import Retriever


class FakeEmbeddingClient:
    def embed(self, text: str) -> list[float]:
        return [0.1] * settings.embedding_size


def test_ingestion_and_retrieval_work_with_real_qdrant():
    embedding_client = FakeEmbeddingClient()

    qdrant_repository = QdrantRepository(
        collection_name="documents_test"
    )

    qdrant_repository.ensure_collection()

    ingestion_service = IngestionService(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    chunk = DocumentChunk(
        id="integration-test-chunk",
        document_id=1,
        chunk_index=0,
        text="Qdrant is a vector database.",
        source="integration-test.txt",
        metadata={},
    )

    ingestion_service.ingest(chunk=chunk)

    assert qdrant_repository.count() == 1

    retriever = Retriever(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    results = retriever.search(
        query="What is Qdrant?",
        limit=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.document.id == chunk.id
    assert result.document.document_id == chunk.document_id
    assert result.document.text == chunk.text
