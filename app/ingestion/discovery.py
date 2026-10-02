from pathlib import Path


class DocumentDiscovery:
    def __init__(self, documents_dir: Path) -> None:
        self.documents_dir = documents_dir

    def find(self) -> list[Path]:
        return sorted(self.documents_dir.glob("*.txt"))
