from app.retrieval.lexical_retrieval import LexicalRetriever
from app.core.models import DocumentChunk


def test_lexical_retriever_returns_chunks_matching_query():
    chunks = [
        DocumentChunk(
            id="1_0",
            document_id=1,
            chunk_index=0,
            text="Qdrant is a vector database.",
            source="qdrant.txt",
            metadata={},
        ),
        DocumentChunk(
            id="2_0",
            document_id=2,
            chunk_index=0,
            text="Python is a programming language.",
            source="python.txt",
            metadata={},
        ),
        DocumentChunk(
            id="3_0",
            document_id=3,
            chunk_index=0,
            text="Qdrant supports vector search.",
            source="search.txt",
            metadata={},
        ),
    ]

    retriever = LexicalRetriever(chunks)

    results = retriever.search(
        "Qdrant is a vector database",
        limit=2,
    )

    assert len(results) == 2


    assert results[0].document.id == "1_0"
    assert results[1].document.id == "3_0"

    assert results[0].score > 0
    assert results[1].score > 0


def test_lexical_retriever_returns_empty_for_unknown_query():
    chunks = [
        DocumentChunk(
            id="1_0",
            document_id=1,
            chunk_index=0,
            text="Qdrant is a vector database.",
            source="qdrant.txt",
            metadata={},
        ),
    ]

    retriever = LexicalRetriever(chunks)

    results = retriever.search(
        "quantum computing",
        limit=5,
    )

    assert results == []
