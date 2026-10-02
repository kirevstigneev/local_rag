from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]


class Document(BaseModel):
    id: int
    text: str
    source: str
    metadata: dict


class DocumentChunk(BaseModel):
    id: str
    document_id: int
    chunk_index: int
    text: str
    source: str
    metadata: dict


class SearchResult(BaseModel):
    score: float
    document: DocumentChunk
