from sentence_transformers import CrossEncoder

from app.core.models import SearchResult


class Reranker:
    def __init__(
            self,
            model_name: str,
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [
            [query, result.document.text]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = [
            SearchResult(
                score=float(score),
                document=result.document,
            )
            for result, score in zip(results, scores)
        ]

        reranked.sort(
            key=lambda result: result.score,
            reverse=True
        )

        return reranked[:limit]
