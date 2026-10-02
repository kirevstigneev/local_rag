from rank_bm25 import BM25Okapi

from app.core.models import DocumentChunk, SearchResult


class LexicalRetriever:
    def __init__(
            self,
            chunks: list[DocumentChunk],
    ) -> None:
        self.chunks = chunks

        tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(
            self,
            query: str,
            limit: int = 5
    ) -> list[SearchResult]:
        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            SearchResult(
                score=float(score),
                document=chunk,
            )
            for chunk, score in ranked[:limit]
            if score > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
