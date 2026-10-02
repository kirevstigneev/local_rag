from unittest.mock import Mock

from app.retrieval.hybrid_retrieval import HybridRetriever
from app.core.models import DocumentChunk, SearchResult


def make_result(
        chunk_id: str,
        score: float,
) -> SearchResult:
    return SearchResult(
        score=score,
        document=DocumentChunk(
            id=chunk_id,
            document_id=1,
            chunk_index=0,
            text=f"Chunk {chunk_id}",
            source="test.txt",
            metadata={},
        ),
    )


def test_hybrid_retriever_combines_vector_and_lexical_results():
    vector_retriever = Mock()
    lexical_retriever = Mock()

    chunk_a = make_result("a", 0.90)
    chunk_b = make_result("b", 0.80)
    chunk_c = make_result("c", 0.70)

    vector_retriever.search.return_value = [
        chunk_a,
        chunk_b,
        chunk_c,
    ]

    lexical_retriever.search.return_value = [
        chunk_c,
        chunk_a,
        chunk_b
    ]

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lexical_retriever=lexical_retriever
    )

    results = retriever.search(
        "Qdrant",
        limit=3
    )

    assert len(results) == 3

    assert {result.document.id for result in results} == {
        "a",
        "b",
        "c",
    }

    assert results[0].document.id == "a"
    assert results[1].document.id == "c"
    assert results[2].document.id == "b"


def test_hybrid_retriever_does_not_duplicate_same_chunk():
    vector_retriever = Mock()
    lexical_retriever = Mock()

    chunk_a_vector = make_result("a", 0.90)
    chunk_a_lexical = make_result("a", 5.00)

    vector_retriever.search.return_value = [
        chunk_a_vector,
    ]

    lexical_retriever.search.return_value = [
        chunk_a_lexical,
    ]

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lexical_retriever=lexical_retriever,
    )

    results = retriever.search(
        "Qdrant",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].document.id == "a"
