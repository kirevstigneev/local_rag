from unittest.mock import Mock

from app.core.models import DocumentChunk, RAGResponse, SearchResult
from app.rag.rag import RAGService


def test_rag_retrieves_chunks_and_generates_answer():
    retriever = Mock()
    reranker = Mock()
    llm_client = Mock()
    context_builder = Mock()

    question = "What is Qdrant?"

    search_results = [
        SearchResult(
            score=0.95,
            document=DocumentChunk(
                id="1_0",
                document_id=1,
                chunk_index=0,
                text="Qdrant is a vector database.",
                source="qdrant.txt",
                metadata={},
            ),
        ),
        SearchResult(
            score=0.90,
            document=DocumentChunk(
                id="2_1",
                document_id=1,
                chunk_index=1,
                text="Qdrant supports semantic search.",
                source="qdrant.txt",
                metadata={},
            ),
        ),
    ]

    reranked_results = [
        SearchResult(
            score=0.99,
            document=search_results[1].document,
        ),
        SearchResult(
            score=0.80,
            document=search_results[0].document
        ),
    ]

    retriever.search.return_value = search_results
    reranker.rerank.return_value = reranked_results

    context = (
        "[Source: qdrant.txt]\n"
        "Qdrant supports semantic search."
        "[Source: qdrant.txt]\n"
        "Qdrant is a vector database.\n\n"
    )

    context_builder.build.return_value = context

    llm_client.generate.return_value = (
        "Qdrant is a vector database."
    )

    service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
        context_builder=context_builder,
        reranker=reranker,
    )

    result = service.answer(question)

    assert result == RAGResponse(
        answer="Qdrant is a vector database.",
        sources=[
            "qdrant.txt",
            "qdrant.txt",
        ],
    )

    retriever.search.assert_called_once_with(
        question,
        limit=5,
    )

    reranker.rerank.assert_called_once_with(
        query=question,
        results=search_results,
        limit=5,
    )

    context_builder.build.assert_called_once_with(
        reranked_results
    )

    expected_prompt = (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    llm_client.generate.assert_called_once_with(
        expected_prompt
    )


def test_rag_returns_fallback_when_no_relevant_results():
    retriever = Mock()
    reranker = Mock()
    llm_client = Mock()
    context_builder = Mock()

    question = "What is quantum computing?"

    retriever.search.return_value = []

    service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
        context_builder=context_builder,
        reranker=reranker
    )

    result = service.answer(question)

    assert result.answer == (
        "I don't have enough information "
        "in the provided documents."
    )

    assert result.sources == []

    llm_client.generate.assert_not_called()
    context_builder.build.assert_not_called()
    reranker.rerank.assert_not_called()

    retriever.search.assert_called_once_with(
        question,
        limit=5,
    )
