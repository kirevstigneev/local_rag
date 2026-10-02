from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str
    qdrant_grpc_host: str
    qdrant_grpc_port: int

    ollama_url: str

    embedding_model: str
    llm_model: str

    qdrant_collection: str
    embedding_size: int

    chunk_size_words: int
    chunk_overlap_words: int
    chunk_min_size_words: int

    documents_dir: Path

    retrieval_limit: int
    retrieval_min_score: float

    reranker_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()
