import json
from pathlib import Path

from app.evaluation.evaluation import EvaluationCase
from app.evaluation.evaluation_dataset import EvaluationDataset


def test_evaluation_dataset_loads_cases_from_json(tmp_path):
    cases_file = tmp_path / "cases.json"

    cases_file.write_text(
        json.dumps(
            [
                {
                    "question": "What is Qdrant?",
                    "expected_answer": "Qdrant is a vector database.",
                    "expected_source": "test.txt",
                },
                {
                    "question": "What does Qdrant enable?",
                    "expected_answer": "semantic search",
                    "expected_source": "test.txt",
                },
            ]
        ),
        encoding="utf-8",
    )


    dataset = EvaluationDataset(cases_file)

    cases = dataset.load()

    assert cases == [
        EvaluationCase(
            question="What is Qdrant?",
            expected_answer="Qdrant is a vector database.",
            expected_source="test.txt",
        ),
        EvaluationCase(
            question="What does Qdrant enable?",
            expected_answer="semantic search",
            expected_source="test.txt",
        ),
    ]


def test_evaluation_dataset_loads_real_dataset():
    cases_file = Path("data/evaluation/cases.json")

    dataset = EvaluationDataset(cases_file)

    cases = dataset.load()

    assert len(cases) == 5

    assert cases[0] == EvaluationCase(
        question="What is Qdrant?",
        expected_answer="Qdrant is a vector database.",
        expected_source="test.txt",
    )
