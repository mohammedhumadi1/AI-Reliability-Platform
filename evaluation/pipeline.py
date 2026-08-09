from dataclasses import dataclass

from evaluation.evaluators.correctness import (
    CorrectnessEvaluator,
)
from evaluation.evaluators.faithfulness import (
    FaithfulnessEvaluator,
)
from evaluation.evaluators.hallucination import (
    HallucinationRiskEvaluator,
)
from evaluation.evaluators.numeric import (
    NumericConsistencyEvaluator,
)
from evaluation.evaluators.status import (
    StatusClassifier,
)


@dataclass(frozen=True)
class EvaluationPipelineResult:
    correctness_score: float
    faithfulness_score: float
    hallucination_risk: float
    status: str
    explanation: str


class EvaluationPipeline:
    CONTRADICTION_MAX_SCORE = 0.20

    def __init__(self) -> None:
        self.correctness_evaluator = (
            CorrectnessEvaluator()
        )

        self.faithfulness_evaluator = (
            FaithfulnessEvaluator()
        )

        self.numeric_evaluator = (
            NumericConsistencyEvaluator()
        )

        self.hallucination_evaluator = (
            HallucinationRiskEvaluator()
        )

        self.status_classifier = (
            StatusClassifier()
        )

    def evaluate(
        self,
        answer: str,
        contexts: list[str],
        reference_answer: str | None = None,
    ) -> EvaluationPipelineResult:
        clean_answer = answer.strip()

        if not clean_answer:
            raise ValueError(
                "The generated answer cannot be empty."
            )

        clean_contexts = [
            context.strip()
            for context in contexts
            if context and context.strip()
        ]

        combined_context = "\n\n".join(
            clean_contexts
        )

        clean_reference = (
            reference_answer.strip()
            if (
                reference_answer
                and reference_answer.strip()
            )
            else None
        )

        correctness = (
            self.correctness_evaluator.evaluate(
                answer=clean_answer,
                reference_answer=clean_reference,
                fallback_context=combined_context,
            )
        )

        faithfulness = (
            self.faithfulness_evaluator.evaluate(
                answer=clean_answer,
                contexts=clean_contexts,
            )
        )

        evidence = (
            clean_reference
            or combined_context
        )

        numeric = (
            self.numeric_evaluator.evaluate(
                answer=clean_answer,
                evidence=evidence,
            )
        )

        correctness_score = correctness.score
        faithfulness_score = faithfulness.score

        if numeric.contradiction:
            correctness_score = min(
                correctness_score,
                self.CONTRADICTION_MAX_SCORE,
            )

            faithfulness_score = min(
                faithfulness_score,
                self.CONTRADICTION_MAX_SCORE,
            )

        hallucination = (
            self.hallucination_evaluator.evaluate(
                faithfulness_score=(
                    faithfulness_score
                ),
                numeric_contradiction=(
                    numeric.contradiction
                ),
            )
        )

        status = self.status_classifier.classify(
            correctness_score=correctness_score,
            faithfulness_score=faithfulness_score,
            numeric_contradiction=(
                numeric.contradiction
            ),
        )

        explanation_parts = [
            status.explanation,
            correctness.explanation,
            faithfulness.explanation,
            numeric.explanation,
            hallucination.explanation,
        ]

        return EvaluationPipelineResult(
            correctness_score=round(
                correctness_score,
                4,
            ),
            faithfulness_score=round(
                faithfulness_score,
                4,
            ),
            hallucination_risk=(
                hallucination.score
            ),
            status=status.status,
            explanation=" ".join(
                explanation_parts
            ),
        )


evaluation_pipeline = EvaluationPipeline()


def run_evaluation(
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
) -> EvaluationPipelineResult:
    return evaluation_pipeline.evaluate(
        answer=answer,
        contexts=contexts,
        reference_answer=reference_answer,
    )