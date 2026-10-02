from app.core.models import SearchResult


class HybridRetriever:
    def __init__(
            self,
            vector_retriever,
            lexical_retriever,
            rrf_k: int = 60,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k

    def search(
            self,
            query: str,
            limit: int = 5,
    ) -> list[SearchResult]:
        vector_results = self.vector_retriever.search(
            query,
            limit=limit,
        )

        lexical_results = self.lexical_retriever.search(
            query,
            limit=limit,
        )

        scores = {}
        results = {}

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.document.id

            scores[chunk_id] = scores.get(chunk_id, 0.0)
            scores[chunk_id] += 1.0 / (
                self.rrf_k + rank
            )

            results[chunk_id] = result

        for rank, result in enumerate(lexical_results, start=1):
            chunk_id = result.document.id

            scores[chunk_id] = scores.get(chunk_id, 0.0)
            scores[chunk_id] += 1.0 / (
                self.rrf_k + rank
            )

            results[chunk_id] = result

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        return [
            SearchResult(
                score=scores[chunk_id],
                document=results[chunk_id].document,
            )
            for chunk_id in ranked_ids[:limit]
        ]
