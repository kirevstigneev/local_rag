import pytest

from app.ingestion.chunking import TextChunker
from app.core.models import Document


@pytest.fixture
def document():
    return Document(
        id=1,
        text="one two three four five six seven eight nine ten",
        source="text.txt",
        metadata={"category": "text"},
    )


def test_split_creates_chunks(document):
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
        min_chunk_size=2,
    )

    chunks = chunker.split(document)

    assert len(chunks) == 3

    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six seven"
    assert chunks[2].text == "seven eight nine ten"


def test_does_not_create_tiny_last_chunk(document):
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
        min_chunk_size=2,
    )

    chunks = chunker.split(document)

    assert len(chunks) == 3

    assert all(
        len(chunk.text.split()) >= 2
        for chunk in chunks
    )


def test_chunks_keep_document_reference(document):
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
        min_chunk_size=2,
    )

    chunks = chunker.split(document)

    assert all(
        chunk.document_id == document.id
        for chunk in chunks
    )


def test_chunks_have_sequential_indexes(document):
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
        min_chunk_size=2,
    )

    chunks = chunker.split(document)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_chunks_keep_document_metadata(document):
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
        min_chunk_size=2,
    )

    chunks = chunker.split(document)

    assert all(
        chunk.metadata == document.metadata
        for chunk in chunks
    )


def test_chunk_ids_are_deterministic(document):
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
        min_chunk_size=2,
    )

    first = chunker.split(document)
    second = chunker.split(document)

    assert [chunk.id for chunk in first] == [
        chunk.id for chunk in second
    ]


def test_empty_document_returns_no_chunks():
    document = Document(
        id=1,
        text="",
        source="empty.txt",
        metadata={},
    )

    chunker = TextChunker()

    assert chunker.split(document) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=10,
            overlap=10,
            min_chunk_size=2,
        )


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=0,
            overlap=0,
            min_chunk_size=2
        )


def test_min_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=10,
            overlap=2,
            min_chunk_size=0
        )


def test_min_chunk_size_must_not_exceed_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=10,
            overlap=2,
            min_chunk_size=11
        )
