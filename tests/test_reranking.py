from unittest.mock import Mock

from app.core.models import DocumentChunk, SearchResult
from app.retrieval.reranking import Reranker


def make_result(
        chunk_id: str,
        text: str,
        score: float,
) -> SearchResult:
    return SearchResult(
        score=score,
        document=DocumentChunk(
            id=chunk_id,
            document_id=1,
            chunk_index=0,
            text=text,
            source="test.txt",
            metadata={},
        ),
    )


def test_reranker_reorders_results_by_cross_encoder_score():
    model = Mock()

    model.predict.return_value = [
        0.20,
        0.95,
        0.60,
    ]

    reranker = Reranker.__new__(Reranker)
    reranker.model = model

    results = [
        make_result("a", "Qdrant is a vector database.", 0.90),
        make_result("b", "Qdrant supports semantic search.", 0.80),
        make_result("c", "Python is a programming language.", 0.70),
    ]

    reranked = reranker.rerank(
        query="How does Qdrant perform semantic search?",
        results=results,
        limit=3
    )

    assert [result.document.id for result in reranked] == [
        "b", "c", "a"
    ]

    assert [result.score for result in reranked] == [
        0.95, 0.60, 0.20
    ]

    model.predict.assert_called_once_with(
        [
            [
                "How does Qdrant perform semantic search?",
                "Qdrant is a vector database.",
            ],
            [
                "How does Qdrant perform semantic search?",
                "Qdrant supports semantic search.",
            ],
            [
                "How does Qdrant perform semantic search?",
                "Python is a programming language.",
            ],
        ]
    )


def test_reranker_limits_results():
    model = Mock()

    model.predict.return_value = [
        0.20, 0.95, 0.60
    ]

    reranker = Reranker.__new__(Reranker)
    reranker.model = model

    results = [
        make_result("a", "First chunk.", 0.90),
        make_result("b", "Second chunk.", 0.80),
        make_result("c", "Third chunk.", 0.70),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
        limit=2
    )

    assert [result.document.id for result in reranked] == [
        "b", "c"
    ]


def test_reranker_returns_empty_for_empty_results():
    model = Mock()

    reranker = Reranker.__new__(Reranker)
    reranker.model = model

    results = reranker.rerank(
        query="Qdrant",
        results=[],
        limit=5,
    )

    assert results == []

    model.predict.assert_not_called()
