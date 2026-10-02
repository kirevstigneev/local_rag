
from pathlib import Path
import re

from pydantic import BaseModel

from app.rag.rag import RAGService


class AnswerEvaluation(BaseModel):
    passed: bool
    reason: str


class EvaluationCase(BaseModel):
    question: str
    expected_answer: str
    expected_source: str | None


class EvaluationResult(BaseModel):
    question: str
    retrieval_passed: bool
    answer_passed: bool


class EvaluationReport(BaseModel):
    total: int
    passed: int
    failed: int
    retrieval_accuracy: float
    answer_accuracy: float
    overall_accuracy: float
    results: list[EvaluationResult]


def _source_matches(expected: str, actual_sources: list[str]) -> bool:
    expected_path = Path(expected)

    return any(
        Path(source).name == expected_path.name
        for source in actual_sources
    )


class AnswerEvaluator:
    def __init__(self, judge) -> None:
        self.judge = judge

    def evaluate(
            self,
            question: str,
            expected_answer: str,
            actual_answer: str
    ) -> AnswerEvaluation:
        return self.judge.evaluate(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer
        )


class EvaluationRunner:
    def __init__(
            self,
            rag_service: RAGService,
            answer_evaluator: AnswerEvaluator,
    ) -> None:
        self.rag_service = rag_service
        self.answer_evaluator = answer_evaluator

    def evaluate(
            self,
            cases: list[EvaluationCase],
    ) -> EvaluationReport:
        results = []

        for evaluation_case  in cases:
            response = self.rag_service.answer(
                evaluation_case.question
            )

            if evaluation_case.expected_source is None:
                retrieval_passed = not response.sources
            else:
                retrieval_passed = _source_matches(
                    evaluation_case.expected_source,
                    response.sources,
                )

            answer_evaluation = self.answer_evaluator.evaluate(
                question=evaluation_case.question,
                expected_answer=evaluation_case.expected_answer,
                actual_answer=response.answer,
            )

            answer_passed = answer_evaluation.passed

            print()
            print("=" * 60)
            print(f"Question: {evaluation_case.question}")
            print(f"Expected answer: {evaluation_case.expected_answer}")
            print(f"Actual answer: {response.answer}")
            print(f"Expected source: {evaluation_case.expected_source}")
            print(f"Actual sources: {response.sources}")
            print(f"Retrieval: {'PASS' if retrieval_passed else 'FAIL'}")
            print(f"Answer: {'PASS' if answer_passed else 'FAIL'}")

            results.append(
                EvaluationResult(
                    question=evaluation_case.question,
                    retrieval_passed=retrieval_passed,
                    answer_passed=answer_passed
                )
            )

        passed = sum(
            result.retrieval_passed
            and result.answer_passed
            for result in results
        )

        retrieval_passed_count = sum(
            result.retrieval_passed
            for result in results
        )

        answer_passed_count = sum(
            result.answer_passed
            for result in results
        )

        if not results:
            retrieval_accuracy = 0
            answer_accuracy = 0
            overall_accuracy = 0
        else:
            retrieval_accuracy = retrieval_passed_count / len(results)
            answer_accuracy = answer_passed_count / len(results)
            overall_accuracy = passed / len(results)

        return EvaluationReport(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            retrieval_accuracy=retrieval_accuracy,
            answer_accuracy=answer_accuracy,
            overall_accuracy=overall_accuracy,
            results=results,
        )
