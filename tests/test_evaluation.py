from unittest.mock import Mock

from app.evaluation.evaluation import (
    AnswerEvaluation,
    AnswerEvaluator,
    EvaluationCase,
    EvaluationRunner
)  
from app.core.models import RAGResponse


def test_evaluation_runner_evaluates_rag_response():
    rag_service = Mock()

    rag_service.answer.return_value = RAGResponse(
        answer="Qdrant is a vector database.",
        sources=["qdrant.txt"],
    )

    answer_evaluator = Mock()

    answer_evaluator.evaluate.return_value = AnswerEvaluation(
        passed=True,
        reason="Answer is correct.",
    )

    runner = EvaluationRunner(
        rag_service=rag_service,
        answer_evaluator=answer_evaluator,
    )

    cases = [
        EvaluationCase(
            question="What is Qdrant?",
            expected_answer="Qdrant is a vector database.",
            expected_source="qdrant.txt",
        )
    ]

    report = runner.evaluate(cases)

    assert report.total == 1
    assert report.passed == 1
    assert report.failed == 0

    assert report.results[0].question == "What is Qdrant?"
    assert report.results[0].retrieval_passed is True
    assert report.results[0].answer_passed is True

    rag_service.answer.assert_called_once_with(
        "What is Qdrant?"
    )


def test_evaluation_runner_reports_failed_case():
    rag_service = Mock()

    rag_service.answer.return_value = RAGResponse(
        answer="I don't know.",
        sources=["other.txt"],
    )

    answer_evaluator = Mock()

    answer_evaluator.evaluate.return_value = AnswerEvaluation(
        passed=False,
        reason="Answer is incorrect.",
    )

    runner = EvaluationRunner(
        rag_service=rag_service,
        answer_evaluator=answer_evaluator,
    )

    cases = [
        EvaluationCase(
            question="What is Qdrant?",
            expected_answer="Qdrant is a vector database.",
            expected_source="qdrant.txt",
        )
    ]

    report = runner.evaluate(cases)

    assert report.total == 1
    assert report.passed == 0
    assert report.failed == 1

    assert report.results[0].retrieval_passed is False
    assert report.results[0].answer_passed is False


def test_evaluation_runner_calculates_accuracy_metrics():
    rag_service = Mock()

    rag_service.answer.side_effect = [
        RAGResponse(
            answer="Qdrant is a vector database.",
            sources=["qdrant.txt"],
        ),
        RAGResponse(
            answer="I don't know.",
            sources=["qdrant.txt"],
        ),
        RAGResponse(
            answer="Semantic search",
            sources=["other.txt"]
        ),
    ]

    answer_evaluator = Mock()

    answer_evaluator.evaluate.side_effect = [
        AnswerEvaluation(
            passed=True,
            reason="Answer is correct.",
        ),
        AnswerEvaluation(
            passed=False,
            reason="Answer is incorrect.",
        ),
        AnswerEvaluation(
            passed=True,
            reason="Answer is correct.",
        ),
    ]

    runner = EvaluationRunner(
        rag_service=rag_service,
        answer_evaluator=answer_evaluator,
    )

    cases = [
        EvaluationCase(
            question="What is Qdrant?",
            expected_answer="Qdrant is a vector database.",
            expected_source="qdrant.txt",
        ),
        EvaluationCase(
            question="What is Qdrant used for?",
            expected_answer="vector search",
            expected_source="qdrant.txt",
        ),
        EvaluationCase(
            question="What does Qdrant enable?",
            expected_answer="semantic search",
            expected_source="qdrant.txt",
        ),
    ]

    report = runner.evaluate(cases)

    assert report.total == 3
    assert report.passed == 1
    assert report.failed == 2

    assert report.retrieval_accuracy == 2/3
    assert report.answer_accuracy == 2/3
    assert report.overall_accuracy == 1/3


def test_evaluation_runner_handles_empty_cases():
    rag_service = Mock()

    answer_evaluator = Mock()

    runner = EvaluationRunner(
        rag_service=rag_service,
        answer_evaluator=answer_evaluator,
    )

    report = runner.evaluate([])

    assert report.total == 0
    assert report.passed == 0
    assert report.failed == 0

    assert report.retrieval_accuracy == 0
    assert report.answer_accuracy == 0
    assert report.overall_accuracy == 0

    rag_service.answer.assert_not_called()


def test_evaluation_runner_uses_answer_evaluator():
    rag_service = Mock()
    answer_evaluator = Mock()

    rag_service.answer.return_value = RAGResponse(
        answer="Qdrant does not support semantic search.",
        sources=["qdrant.txt"],
    )

    answer_evaluator.evaluate.return_value = AnswerEvaluation(
        passed=False,
        reason="The answer contradicts the expected answer.",
    )

    runner = EvaluationRunner(
        rag_service=rag_service,
        answer_evaluator=answer_evaluator,
    )

    cases = [
        EvaluationCase(
            question="What does Qdrant enable?",
            expected_answer="semantic search",
            expected_source="qdrant.txt",
        )
    ]

    report = runner.evaluate(cases)

    answer_evaluator.evaluate.assert_called_once_with(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant does not support semantic search.",
    )

    assert report.results[0].answer_passed is False


def test_answer_evaluator_accepts_semantically_correct_answer():
    judge = Mock()
    judge.evaluate.return_value = AnswerEvaluation(
        passed=True,
        reason="The answer correctly states that Qdrant enables semantic search.",
    )

    evaluator = AnswerEvaluator(judge)

    result = evaluator.evaluate(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant enables semantic search.",
    )

    assert result.passed is True
    assert "semantic search" in result.reason

    judge.evaluate.assert_called_once_with(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant enables semantic search.",
    )


def test_answer_evaluator_rejects_semantically_contradictory_answer():
    judge = Mock()
    judge.evaluate.return_value = AnswerEvaluation(
        passed=False,
        reason="The answer contradicts the expected answer.",
    )

    evaluator = AnswerEvaluator(judge)

    result = evaluator.evaluate(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant does not support semantic search.",
    )

    assert result.passed is False
    assert "contradicts" in result.reason

    judge.evaluate.assert_called_once_with(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant does not support semantic search.",
    )

def test_answer_evaluator_delegates_to_judge():
    judge = Mock()

    judge.evaluate.return_value = AnswerEvaluation(
        passed=True,
        reason="The actual answer is semantically equivalent.",
    )

    evaluator = AnswerEvaluator(judge)

    result = evaluator.evaluate(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant supports semantic search.",
    )

    judge.evaluate.assert_called_once_with(
        question="What does Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant supports semantic search.",
    )

    assert result.passed is True
