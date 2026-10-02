from unittest.mock import Mock

from app.evaluation.evaluation import AnswerEvaluation
from app.evaluation.judge import LLMJudge


def test_llm_judge_accepts_semantically_correct_answer():
    llm_client = Mock()

    llm_client.generate.return_value = """
        {
            "passed": true,
            "reason": "The actual answer is semantically equivalent to the expected answer."
        }
    """

    judge = LLMJudge(llm_client)

    result = judge.evaluate(
        question="What is Qdrant?",
        expected_answer="Qdrant is a vector database.",
        actual_answer="Qdrant is a database for storing vectors."
    )

    assert result == AnswerEvaluation(
        passed=True,
        reason="The actual answer is semantically equivalent to the expected answer.",
    )


def test_llm_judge_rejects_semantically_contradictory_answer():
    llm_client = Mock()

    llm_client.generate.return_value = """
        {
            "passed": false,
            "reason": "The actual answer contradicts the expected answer."
        }
    """

    judge = LLMJudge(llm_client)

    result = judge.evaluate(
        question="What is Qdrant enable?",
        expected_answer="semantic search",
        actual_answer="Qdrant does not supprt semantic search.",
    )

    assert result == AnswerEvaluation(
        passed=False,
        reason="The actual answer contradicts the expected answer.",
    )


def test_llm_judge_sends_question_and_answers_to_llm():
    llm_client = Mock()

    llm_client.generate.return_value = """
        {
            "passed": true,
            "reason": "The answers express the same meaning."
        }
    """

    judge = LLMJudge(llm_client)

    judge.evaluate(
        question="What is Qdrant?",
        expected_answer="Qdrant is a vector database.",
        actual_answer="Qdrant stores vectors in a database.",
    )

    llm_client.generate.assert_called_once()

    prompt = llm_client.generate.call_args.args[0]

    assert "What is Qdrant?" in prompt
    assert "Qdrant is a vector database." in prompt
    assert "Qdrant stores vectors in a database." in prompt
