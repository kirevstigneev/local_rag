from app.rag.context import ContextBuilder
from app.core.models import DocumentChunk, SearchResult


def test_context_builder_includes_sources_and_text():
    builder = ContextBuilder()

    results = [
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
                id="2_0",
                document_id=2,
                chunk_index=0,
                text="Qdrant supports semantic search.",
                source="search.txt",
                metadata={},
            ),
        ),
    ]

    context = builder.build(results)

    expected = (
        "[Source: qdrant.txt]\n"
        "Qdrant is a vector database.\n\n"
        "[Source: search.txt]\n"
        "Qdrant supports semantic search."
    )

    assert context == expected


def test_context_builder_returns_empty_string_for_no_results():
    builder = ContextBuilder()

    context = builder.build([])

    assert context == ""
