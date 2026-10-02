import pytest
from unittest.mock import Mock

from app.ingestion.ingestion import IngestionService
from app.core.models import DocumentChunk


def test_ingest_embeds_chunk_and_stores_it():
    embedding_client = Mock()
    qdrant_repository = Mock()

    chunk = DocumentChunk(
        id="1_0",
        document_id=1,
        chunk_index=0,
        text="Qdrant stores vectors.",
        source="docs.txt",
        metadata={},
    )

    vector = [0.1, 0.2, 0.3]

    embedding_client.embed.return_value = vector

    service = IngestionService(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    service.ingest(chunk=chunk)

    embedding_client.embed.assert_called_once_with(
        chunk.text,
    )

    qdrant_repository.upsert.assert_called_once_with(
        chunk=chunk,
        vector=vector,
    )


def test_ingest_does_not_store_chunk_when_embedding_fails():
    embedding_client = Mock()
    qdrant_repository = Mock()

    chunk = DocumentChunk(
        id="1_0",
        document_id=1,
        chunk_index=0,
        text="Qdrant stores vectors.",
        source="docs.txt",
        metadata={},
    )

    embedding_client.embed.side_effect = RuntimeError(
        "Embedding service unavailable"
    )

    service = IngestionService(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service unavailable",
    ):
        service.ingest(chunk=chunk)

    embedding_client.embed.assert_called_once_with(
        chunk.text,
    )

    qdrant_repository.upsert.assert_not_called()
    