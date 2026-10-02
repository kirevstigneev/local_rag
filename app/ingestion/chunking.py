import hashlib

from app.core.config import settings
from app.core.models import Document, DocumentChunk


class TextChunker:
    def __init__(
            self,
            chunk_size: int | None = None,
            overlap: int | None = None,
            min_chunk_size: int | None = None,
    ) -> None:
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.chunk_size_words
        )

        self.overlap = (
            overlap
            if overlap is not None
            else settings.chunk_overlap_words
        )

        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else settings.chunk_min_size_words
        )

        if self.overlap < 0:
            raise ValueError(
                "overlap must not be negative"
            )

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        if self.min_chunk_size <= 0:
            raise ValueError(
                "min_chunk_size must be greater than 0"
            )

        if self.min_chunk_size > self.chunk_size:
            raise ValueError(
                "min_chunk_size must not be greater than chunk_size"
            )

    def split(self, document: Document) -> list[DocumentChunk]:
        words = document.text.split()

        if not words:
            return []

        step = self.chunk_size - self.overlap

        chunks: list[DocumentChunk] = []

        for chunk_index, start in enumerate(
            range(0, len(words), step)
        ):
            chunk_words = words[start:start + self.chunk_size]

            if len(chunk_words) < self.min_chunk_size:
                break

            text = " ".join(chunk_words)

            chunk_id = hashlib.sha256(
                f"{document.id}:{chunk_index}:{text}".encode("utf-8") 
            ).hexdigest()

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=text,
                    chunk_index=chunk_index,
                    source=document.source,
                    metadata=document.metadata
                )
            )

        return chunks
