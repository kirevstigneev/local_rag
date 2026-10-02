import json

from app.evaluation.evaluation import AnswerEvaluation


class LLMJudge:
    def __init__(
            self,
            llm_client
    ) -> None:
        self.llm_client = llm_client

    def evaluate(
            self,
            question: str,
            expected_answer: str,
            actual_answer: str,
    ) -> AnswerEvaluation:
        prompt = f"""
            Evaluate whether the actual answer is semantically correct
            compared to the expected asnwer.

            Question:
            {question}

            Expected answer:
            {expected_answer}

            Actual answer:
            {actual_answer}

            Return JSON only in the following format:
            {{
                "passed": true,
                "reason": "short explanation"
            }}
        """

        response = self.llm_client.generate(prompt)
        data = json.loads(response)

        return AnswerEvaluation(**data)
