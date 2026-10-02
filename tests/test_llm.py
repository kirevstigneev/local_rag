import pytest
import requests

from unittest.mock import Mock

from app.core.config import settings
from app.clients.llm import LLMClient

def test_generate_returns_llm_response():
    session = Mock()

    response = Mock()

    response.json.return_value = {
        "message": {
            "content": "Qdrant is a vector database."
        }
    }

    session.post.return_value = response

    client = LLMClient(session=session)

    prompt = "What is Qdrant?"

    result = client.generate(prompt)

    assert result == "Qdrant is a vector database."

    session.post.assert_called_once_with(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status.assert_called_once()


def test_generate_raises_for_http_error():
    session = Mock()

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError(
        "Ollama unavailable"
    )

    session.post.return_value = response

    client = LLMClient(session=session)

    with pytest.raises(requests.HTTPError):
        client.generate("what is Qdrant?")


def test_generate_raises_for_invalid_responce():
    session = Mock()

    response = Mock()

    response.json.return_value = {
        "unexpected": "responce"
    }

    session.post.return_value = response

    client = LLMClient(session=session)

    with pytest.raises(KeyError):
        client.generate("What is Qdrant?")
    