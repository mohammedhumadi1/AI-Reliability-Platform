from __future__ import annotations

from dataclasses import dataclass

from benchmarks.baseline.comparison import (
    PlatformValueAdd,
    build_platform_value_add,
)
from evaluation.pipeline import (
    EvaluationPipelineResult,
    run_evaluation,
)
from knowledge_base.verification_agent import (
    VerificationResult,
    verify_answer,
)


@dataclass(frozen=True)
class KBAblationDeltas:
    diagnosis_changed: bool
    diagnosis_subcategory_changed: bool
    health_score_delta: int
    health_status_changed: bool
    diagnosis_confidence_delta: float | None
    recommendation_count_delta: int
    recommendations_changed: bool


@dataclass(frozen=True)
class KBVerificationAblationResult:
    evaluation: EvaluationPipelineResult
    without_kb: PlatformValueAdd
    with_kb: PlatformValueAdd
    with_kb_verification: VerificationResult
    deltas: KBAblationDeltas


def build_disabled_kb_verification(
) -> VerificationResult:
    """
    Represent the controlled WITHOUT_KB condition.

    NOT_AVAILABLE intentionally preserves the
    platform's proxy-only root-cause behavior.
    """
    return VerificationResult(
        status="NOT_AVAILABLE",
        evidence_found=False,
        is_supported=None,
        best_match_text="",
        best_match_source="",
        similarity_distance=None,
        question_relevance_score=None,
        answer_support_score=None,
        context_alignment_score=None,
        numeric_contradiction=False,
        explanation=(
            "KB verification disabled for "
            "the controlled ablation condition."
        ),
    )


def _confidence_delta(
    without_kb: float | None,
    with_kb: float | None,
) -> float | None:
    if (
        without_kb is None
        or with_kb is None
    ):
        return None

    return round(
        with_kb - without_kb,
        4,
    )


def run_kb_verification_ablation(
    *,
    project_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    prompt: str | None = None,
    evaluation_fn=run_evaluation,
    verification_fn=verify_answer,
) -> KBVerificationAblationResult:
    """
    Compare reliability outputs with and without
    independent KB verification.

    The core evaluation is executed exactly once.
    Both conditions therefore use identical
    question, answer, contexts, prompt, reference,
    and six core evaluation metrics. The only
    controlled variable is KB verification.
    """
    evaluation = evaluation_fn(
        question=question,
        answer=answer,
        contexts=contexts,
        reference_answer=reference_answer,
    )

    without_verification = (
        build_disabled_kb_verification()
    )

    without_kb = build_platform_value_add(
        result=evaluation,
        verification=without_verification,
        prompt=prompt,
    )

    with_verification = verification_fn(
        project_id=project_id,
        question=question,
        answer=answer,
        rag_contexts=contexts,
    )

    with_kb = build_platform_value_add(
        result=evaluation,
        verification=with_verification,
        prompt=prompt,
    )

    deltas = KBAblationDeltas(
        diagnosis_changed=(
            with_kb.diagnosis_category
            != without_kb.diagnosis_category
        ),
        diagnosis_subcategory_changed=(
            with_kb.diagnosis_subcategory
            != without_kb.diagnosis_subcategory
        ),
        health_score_delta=(
            with_kb.health_score
            - without_kb.health_score
        ),
        health_status_changed=(
            with_kb.health_status
            != without_kb.health_status
        ),
        diagnosis_confidence_delta=(
            _confidence_delta(
                without_kb.diagnosis_confidence,
                with_kb.diagnosis_confidence,
            )
        ),
        recommendation_count_delta=(
            with_kb.recommendation_count
            - without_kb.recommendation_count
        ),
        recommendations_changed=(
            with_kb.recommendation_actions
            != without_kb.recommendation_actions
        ),
    )

    return KBVerificationAblationResult(
        evaluation=evaluation,
        without_kb=without_kb,
        with_kb=with_kb,
        with_kb_verification=with_verification,
        deltas=deltas,
    )
