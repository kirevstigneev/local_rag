from unittest.mock import Mock

from app.core.models import DocumentChunk, SearchResult
from app.retrieval.retrieval import Retriever


def test_retriever_embeds_query_and_searches_qdrant():
    embedding_client = Mock()
    qdrant_repository = Mock()

    query = "How can I search documents by meaning?"
    vector = [0.1, 0.2, 0.3]

    expected_result = [
        SearchResult(
            score=0.95,
            document=DocumentChunk(
                id="1_0",
                document_id=1,
                chunk_index=0,
                text="Qdrant performs semantic search.",
                source="docs.txt",
                metadata={},
            ),
        )
    ]

    embedding_client.embed.return_value = vector
    qdrant_repository.search.return_value = expected_result

    retriever = Retriever(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository
    )

    results = retriever.search(query, limit=5)

    assert results == expected_result

    embedding_client.embed.assert_called_once_with(query)

    qdrant_repository.search.assert_called_once_with(
        vector=vector,
        limit=5
    )


def test_retriever_filters_results_below_min_score():
    embedding_client = Mock()
    qdrant_repository = Mock()

    embedding_client.embed.return_value = [0.1, 0.2, 0.3]

    relevant_result = SearchResult(
        score=0.85,
        document=DocumentChunk(
            id="1_0",
            document_id=1,
            chunk_index=0,
            text="Relevant chunk.",
            source="test.txt",
            metadata={},
        ),
    )

    weak_result = SearchResult(
        score=0.40,
        document=DocumentChunk(
            id="2_0",
            document_id=1,
            chunk_index=1,
            text="Weak chunk.",
            source="test.txt",
            metadata={},
        ),
    )

    qdrant_repository.search.return_value = [
        relevant_result,
        weak_result,
    ]

    retriever = Retriever(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    results = retriever.search("Qdrant")

    assert results == [relevant_result]
