from pathlib import Path

from app.container import create_rag_service
from app.evaluation.evaluation import EvaluationRunner
from app.evaluation.evaluation_dataset import EvaluationDataset


def main() -> None:
    dataset = EvaluationDataset(
        Path("data/evaluation/cases.json")
    )

    cases = dataset.load()

    rag_service = create_rag_service()

    runner = EvaluationRunner(
        rag_service
    )

    report = runner.evaluate(cases)

    print(f"Total: {report.total}\n")
    print(f"Passed: {report.passed}\n")
    print(f"Failed: {report.failed}\n")

    print(f"Retrieval accuracy: {report.retrieval_accuracy:.2%}\n")
    print(f"Answer accuracy: {report.answer_accuracy:.2%}\n")
    print(f"Overall accuracy: {report.overall_accuracy:.2%}\n")


if __name__ == "__main__":
    main()
