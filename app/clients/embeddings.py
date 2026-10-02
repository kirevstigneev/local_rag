import requests

from app.core.config import settings


class EmbeddingClient:
    def __init__(self) -> None:
        self.base_url = settings.ollama_url
        self.model = settings.embedding_model

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        embedding = data.get("embeddings")[0]

        if len(embedding) != settings.embedding_size:
            raise ValueError(
                f"Unexpected embedding size:"
                f"{len(embedding)},"
                f"expected {settings.embedding_size}"
            )

        return embedding
