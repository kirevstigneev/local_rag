import hashlib
from pathlib import Path

from app.core.models import Document


class TextFileLoader:
    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")

        document_id = int(
            hashlib.sha256(
                str(path.resolve()).encode("utf-8")
            ).hexdigest()[:16],
            16,
        )

        return Document(
            id=document_id,
            text=text,
            source=str(path),
            metadata={},
        )
