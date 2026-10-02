from app.core.models import SearchResult


class ContextBuilder:
    def build(self, results: list[SearchResult]) -> str:
        return "\n\n".join(
            f"[Source: {result.document.source}]\n"
            f"{result.document.text}"
            for result in results
        )
