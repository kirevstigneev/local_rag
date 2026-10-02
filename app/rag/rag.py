from app.rag.context import ContextBuilder
from app.clients.llm import LLMClient
from app.core.models import RAGResponse
from app.retrieval.reranking import Reranker
from app.retrieval.retrieval import Retriever


class RAGService:
    def __init__(
            self,
            retriever: Retriever,
            llm_client: LLMClient,
            context_builder: ContextBuilder,
            reranker: Reranker,

    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.reranker = reranker

    def answer(
            self,
            question: str,
            limit: int = 5,
    ) -> RAGResponse:
        results = self.retriever.search(
            question,
            limit=limit,
        )

        if not results:
            return RAGResponse(
                answer="I don't have enough information in the provided documents.",
                sources=[],
            )

        results = self.reranker.rerank(
            query=question,
            results=results,
            limit=limit
        )

        context = self.context_builder.build(results)

        prompt = (
            "Answer the question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

        answer = self.llm_client.generate(prompt)

        sources = [
            result.document.source
            for result in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
        )
