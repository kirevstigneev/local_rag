import uuid

import pytest

from app.core.config import settings
from app.core.models import DocumentChunk
from app.clients.qdrant import QdrantRepository


@pytest.fixture
def first_chunk():
    return DocumentChunk(
        id="1_0",
        document_id=1,
        chunk_index=0,
        text="Python is a programming language",
        source="docs.txt",
        metadata={"category": "programming"},
    )


@pytest.fixture
def second_chunk():
    return DocumentChunk(
        id="2_0",
        document_id=2,
        chunk_index=0,
        text="The weather is sunny today",
        source="weather.txt",
        metadata={"category": "weather"},
    )


@pytest.fixture
def test_chunk():
    return DocumentChunk(
        id="3_0",
        document_id=3,
        chunk_index=0,
        text="First document",
        source="test",
        metadata={},
    )


@pytest.fixture
def qdrant_repository():
    repository = QdrantRepository()

    original_collection_name = repository.collection_name
    test_collection_name = f"test_documents_{uuid.uuid4().hex}"

    repository.collection_name = test_collection_name

    yield repository

    if repository.collection_exists():
        repository.client.delete_collection(
            collection_name=test_collection_name
        )

    repository.collection_name = original_collection_name


def test_ensure_collection_creates_collection(qdrant_repository):
    assert not qdrant_repository.collection_exists()

    qdrant_repository.ensure_collection()

    assert qdrant_repository.collection_exists()


def test_ensure_collection_is_idempotent(qdrant_repository):
    qdrant_repository.ensure_collection()

    assert qdrant_repository.collection_exists()
    assert qdrant_repository.count() == 0

    qdrant_repository.ensure_collection()

    assert qdrant_repository.collection_exists()
    assert qdrant_repository.count() == 0


def test_count_returns_number_of_points(
    qdrant_repository,
    test_chunk,
):
    qdrant_repository.ensure_collection()

    assert qdrant_repository.count() == 0

    qdrant_repository.upsert(
        chunk=test_chunk,
        vector=[0.1] * settings.embedding_size,
    )

    assert qdrant_repository.count() == 1


def test_upsert_adds_point(
    qdrant_repository,
    test_chunk,
):
    qdrant_repository.ensure_collection()

    qdrant_repository.upsert(
        chunk=test_chunk,
        vector=[0.1] * settings.embedding_size,
    )

    assert qdrant_repository.count() == 1


def test_upsert_updates_existing_point(qdrant_repository):
    qdrant_repository.ensure_collection()

    original_chunk = DocumentChunk(
        id="1_0",
        document_id=1,
        chunk_index=0,
        text="Original document",
        source="test",
        metadata={},
    )

    updated_chunk = DocumentChunk(
        id="1_0",
        document_id=1,
        chunk_index=0,
        text="Updated document",
        source="test",
        metadata={},
    )

    qdrant_repository.upsert(
        chunk=original_chunk,
        vector=[0.1] * settings.embedding_size,
    )

    assert qdrant_repository.count() == 1

    qdrant_repository.upsert(
        chunk=updated_chunk,
        vector=[0.2] * settings.embedding_size,
    )

    assert qdrant_repository.count() == 1

    results = qdrant_repository.search(
        vector=[0.2] * settings.embedding_size,
        limit=1,
    )

    assert results[0].document.text == "Updated document"


def test_search_returns_most_similar_point(
    qdrant_repository,
    first_chunk,
    second_chunk,
):
    qdrant_repository.ensure_collection()

    qdrant_repository.upsert(
        chunk=first_chunk,
        vector=[1.0] + [0.0] * (settings.embedding_size - 1),
    )

    qdrant_repository.upsert(
        chunk=second_chunk,
        vector=[0.0, 1.0] + [0.0] * (settings.embedding_size - 2),
    )

    results = qdrant_repository.search(
        vector=[1.0] + [0.0] * (settings.embedding_size - 1),
        limit=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.document.id == first_chunk.id
    assert result.document.document_id == first_chunk.document_id
    assert result.document.chunk_index == first_chunk.chunk_index
    assert result.document.text == first_chunk.text
    assert result.document.source == first_chunk.source
    assert result.document.metadata == first_chunk.metadata
    assert result.score > 0.99


def test_get_all_chunks_returns_stored_chunks(clean_qdrant):
    repository = clean_qdrant

    chunk = DocumentChunk(
        id="1_0",
        document_id=1,
        chunk_index=0,
        text="Qdrant is a vector database.",
        source="test.txt",
        metadata={},
    )

    vector = [0.1] * settings.embedding_size

    repository.upsert(
        chunk=chunk,
        vector=vector,
    )

    chunks = repository.get_all_chunks()

    assert chunks == [chunk]
