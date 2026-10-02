import pytest

from app.clients.qdrant import QdrantRepository


@pytest.fixture
def clean_qdrant():
    repository = QdrantRepository(
        collection_name="documents_test",
    )

    if repository.collection_exists():
        repository.client.delete_collection(
            repository.collection_name
        )

    repository.ensure_collection()

    yield repository

    if repository.collection_exists():
        repository.client.delete_collection(
            repository.collection_name
        )
