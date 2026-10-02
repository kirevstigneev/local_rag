from unittest.mock import Mock

from app.evaluation.evaluation import EvaluationCase, EvaluationReport
from app.evaluation.evaluate import main


def test_evaluation_cli_runs_evaluation(monkeypatch, capsys):
    dataset = Mock()
    runner = Mock()

    cases = [
        EvaluationCase(
            question="Question 1",
            expected_answer="Answer 1",
            expected_source="test.txt",
        ),
        EvaluationCase(
            question="Question 2",
            expected_answer="Answer 2",
            expected_source="test.txt",
        ),
    ]

    dataset.load.return_value = cases

    runner.evaluate.return_value = EvaluationReport(
        total=2,
        passed=2,
        failed=0,
        retrieval_accuracy=1.0,
        answer_accuracy=1.0,
        overall_accuracy=1.0,
        results=[],
    )

    monkeypatch.setattr(
        "app.evaluation.evaluate.EvaluationDataset",
        lambda path: dataset,
    )

    monkeypatch.setattr(
        "app.evaluation.evaluate.EvaluationRunner",
        lambda rag_service: runner,
    )

    monkeypatch.setattr(
        "app.evaluation.evaluate.create_rag_service",
        lambda: Mock(),
    )

    main()

    dataset.load.assert_called_once()

    runner.evaluate.assert_called_once_with(cases)

    output = capsys.readouterr().out

    assert "Total: 2" in output
    assert "Passed: 2" in output
    assert "Failed: 0" in output


def test_evaluation_cli_displays_accuracy_metrics(monkeypatch, capsys):
    dataset = Mock()
    runner = Mock()

    cases = [
        EvaluationCase(
            question="Question 1",
            expected_answer="Answer 1",
            expected_source="test.txt",
        ),
        EvaluationCase(
            question="Question 2",
            expected_answer="Answer 2",
            expected_source="test.txt",
        ),
    ]

    dataset.load.return_value = cases

    runner.evaluate.return_value = EvaluationReport(
        total=2,
        passed=1,
        failed=1,
        retrieval_accuracy=0.5,
        answer_accuracy=1.0,
        overall_accuracy=0.5,
        results=[],
    )

    monkeypatch.setattr(
        "app.evaluation.evaluate.EvaluationDataset",
        lambda path: dataset,
    )

    monkeypatch.setattr(
        "app.evaluation.evaluate.EvaluationRunner",
        lambda rag_service: runner,
    )

    monkeypatch.setattr(
        "app.evaluation.evaluate.create_rag_service",
        lambda: Mock(),
    )

    main()

    output = capsys.readouterr().out

    assert "Total: 2" in output
    assert "Passed: 1" in output
    assert "Failed: 1" in output

    assert "Retrieval accuracy: 50.00%" in output
    assert "Answer accuracy: 100.00%" in output
    assert "Overall accuracy: 50.00%" in output
