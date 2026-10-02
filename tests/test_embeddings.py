from unittest.mock import Mock, patch

import pytest

from app.core.config import settings
from app.clients.embeddings import EmbeddingClient


def test_embed_returns_embedding():
    embedding = [0.1] * settings.embedding_size

    response = Mock()
    response.json.return_value = {
        "embeddings": [embedding]
    }

    with patch("app.clients.embeddings.requests.post", return_value=response) as mock_post:
        client = EmbeddingClient()

        result = client.embed("hello")

    assert result == embedding

    mock_post.assert_called_once_with(
        f"{settings.ollama_url}/api/embed",
        json={
            "model": settings.embedding_model,
            "input": "hello",
        },
        timeout=60,
    )


def test_embed_rejects_unexpected_embedding_size():
    embedding = [0.1, 0.2, 0.3]

    response = Mock()
    response.json.return_value = {
        "embeddings": [embedding]
    }

    with patch("app.clients.embeddings.requests.post", return_value=response):
        client = EmbeddingClient()

        with pytest.raises(ValueError, match="Unexpected embedding size"):
            client.embed("hello")
