from __future__ import annotations

from dataclasses import dataclass

from app.services.health_check_service import (
    build_root_cause_metrics,
)
from evaluation.pipeline import (
    EvaluationPipelineResult,
    run_evaluation,
)
from knowledge_base.verification_agent import (
    VerificationResult,
    verify_answer,
)
from recommendation_engine.engine import (
    generate_recommendations,
)
from reporting.health_score import (
    calculate_health_score,
)
from root_cause.rules.pipeline import (
    run_rules_pipeline,
)


@dataclass(frozen=True)
class CoreMetricSnapshot:
    correctness: float
    faithfulness: float
    context_precision: float
    context_recall: float | None
    answer_relevancy: float
    hallucination_risk: float

    def as_dict(
        self,
    ) -> dict[str, float | None]:
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "context_precision": (
                self.context_precision
            ),
            "context_recall": (
                self.context_recall
            ),
            "answer_relevancy": (
                self.answer_relevancy
            ),
            "hallucination_risk": (
                self.hallucination_risk
            ),
        }


@dataclass(frozen=True)
class PlatformValueAdd:
    verification_status: str
    kb_evidence_found: bool
    knowledge_base_support: float | None
    diagnosis_category: str | None
    diagnosis_subcategory: str | None
    diagnosis_severity: str | None
    diagnosis_confidence: float | None
    health_score: int
    health_status: str
    recommendation_count: int
    recommendation_actions: tuple[str, ...]


@dataclass(frozen=True)
class BaselineComparisonResult:
    base_rag_metrics: CoreMetricSnapshot
    full_platform_metrics: CoreMetricSnapshot
    core_metric_deltas: dict[
        str,
        float | None,
    ]
    platform_value_add: PlatformValueAdd


def _snapshot(
    result: EvaluationPipelineResult,
) -> CoreMetricSnapshot:
    return CoreMetricSnapshot(
        correctness=result.correctness_score,
        faithfulness=result.faithfulness_score,
        context_precision=(
            result.context_precision_score
        ),
        context_recall=(
            result.context_recall_score
        ),
        answer_relevancy=(
            result.answer_relevancy_score
        ),
        hallucination_risk=(
            result.hallucination_risk
        ),
    )


def _metric_deltas(
    base: CoreMetricSnapshot,
    full: CoreMetricSnapshot,
) -> dict[str, float | None]:
    base_values = base.as_dict()
    full_values = full.as_dict()

    deltas: dict[
        str,
        float | None,
    ] = {}

    for name, base_value in (
        base_values.items()
    ):
        full_value = full_values[name]

        if (
            base_value is None
            or full_value is None
        ):
            deltas[name] = None
            continue

        deltas[name] = round(
            full_value - base_value,
            4,
        )

    return deltas


def build_platform_value_add(
    *,
    result: EvaluationPipelineResult,
    verification: VerificationResult,
    prompt: str | None = None,
) -> PlatformValueAdd:
    """
    Calculate reliability value-add for one
    evaluation result and KB verification.

    The helper reuses the same root-cause,
    health-score, and recommendation components
    used by the reliability platform.
    """
    root_cause_metrics = (
        build_root_cause_metrics(
            result=result,
            verification=verification,
            prompt=prompt,
        )
    )

    diagnosis = run_rules_pipeline(
        root_cause_metrics
    )

    health = calculate_health_score(
        {
            "faithfulness": (
                result.faithfulness_score
            ),
            "answer_relevancy": (
                result.answer_relevancy_score
            ),
            "answer_correctness": (
                result.correctness_score
            ),
            "context_precision": (
                result.context_precision_score
            ),
            "context_recall": (
                result.context_recall_score
            ),
            "knowledge_base_support": (
                verification.health_score_component
            ),
        }
    )

    recommendations = (
        generate_recommendations(
            diagnosis
        )
    )

    return PlatformValueAdd(
        verification_status=(
            verification.status
        ),
        kb_evidence_found=(
            verification.evidence_found
        ),
        knowledge_base_support=(
            verification.health_score_component
        ),
        diagnosis_category=(
            diagnosis.get("category")
            if diagnosis
            else None
        ),
        diagnosis_subcategory=(
            diagnosis.get("subcategory")
            if diagnosis
            else None
        ),
        diagnosis_severity=(
            diagnosis.get("severity")
            if diagnosis
            else None
        ),
        diagnosis_confidence=(
            diagnosis.get("confidence")
            if diagnosis
            else None
        ),
        health_score=health.score,
        health_status=health.status,
        recommendation_count=len(
            recommendations
        ),
        recommendation_actions=tuple(
            item.action
            for item in recommendations
        ),
    )


def compare_base_rag_with_platform(
    *,
    project_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    prompt: str | None = None,
    evaluation_fn=run_evaluation,
    verification_fn=verify_answer,
) -> BaselineComparisonResult:
    """
    Compare the same RAG output under the core
    evaluator and the full reliability platform.

    Core metric deltas are expected to be zero
    because both sides evaluate the same output.
    Platform value is represented separately by
    KB verification, root-cause diagnosis, health
    scoring, and recommendations.
    """
    base_result = evaluation_fn(
        question=question,
        answer=answer,
        contexts=contexts,
        reference_answer=reference_answer,
    )

    # The full platform evaluates the exact same
    # RAG output. Reuse the core evaluation rather
    # than running identical metrics twice.
    full_result = base_result

    verification: VerificationResult = (
        verification_fn(
            project_id=project_id,
            question=question,
            answer=answer,
            rag_contexts=contexts,
        )
    )

    platform_value = (
        build_platform_value_add(
            result=full_result,
            verification=verification,
            prompt=prompt,
        )
    )

    base_metrics = _snapshot(
        base_result
    )

    full_metrics = _snapshot(
        full_result
    )

    return BaselineComparisonResult(
        base_rag_metrics=base_metrics,
        full_platform_metrics=full_metrics,
        core_metric_deltas=_metric_deltas(
            base_metrics,
            full_metrics,
        ),
        platform_value_add=platform_value,
    )
