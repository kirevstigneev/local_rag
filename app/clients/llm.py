import requests

from app.core.config import settings


class LLMClient:
    def __init__(self, session=None) -> None:
        self.session = session or requests.Session()

        self.base_url = settings.ollama_url
        self.model = settings.llm_model

    def generate(self, prompt: str) -> str:
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]
