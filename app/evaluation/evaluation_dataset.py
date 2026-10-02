import json
from pathlib import Path

from app.evaluation.evaluation import EvaluationCase


class EvaluationDataset:
    def __init__(
            self,
            path: Path,
        ) -> None:
        self.path = path

    def load(self) -> list[EvaluationCase]:
        data = json.loads(
            self.path.read_text(
                encoding="utf-8",
            )
        )

        return [
            EvaluationCase(**case)
            for case in data
        ]
