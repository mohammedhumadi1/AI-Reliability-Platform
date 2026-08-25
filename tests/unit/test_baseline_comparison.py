from evaluation.pipeline import (
    EvaluationPipelineResult,
)
from knowledge_base.verification_agent import (
    VerificationResult,
)

from benchmarks.baseline.comparison import (
    compare_base_rag_with_platform,
)


def make_evaluation(
    context_recall=0.6,
):
    return EvaluationPipelineResult(
        correctness_score=0.8,
        faithfulness_score=0.7,
        context_precision_score=0.9,
        context_recall_score=(
            context_recall
        ),
        answer_relevancy_score=0.85,
        hallucination_risk=0.3,
        status="GOOD",
        explanation="Example evaluation.",
    )


def make_verification(
    *,
    status="NOT_AVAILABLE",
    context_alignment=None,
    answer_support=None,
):
    return VerificationResult(
        status=status,
        evidence_found=(
            status != "NOT_AVAILABLE"
        ),
        is_supported=(
            True
            if status == "SUPPORTED"
            else (
                False
                if status in {
                    "UNSUPPORTED",
                    "CONTRADICTED",
                }
                else None
            )
        ),
        best_match_text="Company evidence.",
        best_match_source="document.pdf",
        similarity_distance=0.1,
        question_relevance_score=0.9,
        answer_support_score=answer_support,
        context_alignment_score=(
            context_alignment
        ),
        numeric_contradiction=(
            status == "CONTRADICTED"
        ),
        explanation=(
            "Independent verification result."
        ),
    )


def test_same_output_preserves_core_metric_parity():
    calls = []

    def fake_evaluation(**kwargs):
        calls.append(kwargs)
        return make_evaluation()

    result = (
        compare_base_rag_with_platform(
            project_id="project-1",
            question="What is RAG?",
            answer="Example answer.",
            contexts=["Example context."],
            reference_answer=(
                "Reference answer."
            ),
            evaluation_fn=fake_evaluation,
            verification_fn=(
                lambda **kwargs: (
                    make_verification()
                )
            ),
        )
    )

    assert len(calls) == 1
    assert all(
        delta == 0.0
        for delta in (
            result.core_metric_deltas.values()
        )
    )


def test_full_platform_records_verified_retrieval_failure():
    result = (
        compare_base_rag_with_platform(
            project_id="project-1",
            question="What is RAG?",
            answer="Example answer.",
            contexts=["Wrong context."],
            evaluation_fn=(
                lambda **kwargs: (
                    make_evaluation()
                )
            ),
            verification_fn=(
                lambda **kwargs: (
                    make_verification(
                        status="UNSUPPORTED",
                        context_alignment=0.2,
                        answer_support=0.4,
                    )
                )
            ),
        )
    )

    value = result.platform_value_add

    assert (
        value.verification_status
        == "UNSUPPORTED"
    )
    assert (
        value.knowledge_base_support
        == 0.4
    )
    assert (
        value.diagnosis_category
        == "RETRIEVAL_FAILURE"
    )
    assert (
        value.diagnosis_subcategory
        == "VERIFIED_MISSED_EVIDENCE"
    )
    assert value.recommendation_count == 3
    assert len(
        value.recommendation_actions
    ) == 3


def test_missing_kb_is_recorded_without_fake_support():
    result = (
        compare_base_rag_with_platform(
            project_id="project-1",
            question="What is RAG?",
            answer="Example answer.",
            contexts=["Example context."],
            evaluation_fn=(
                lambda **kwargs: (
                    make_evaluation()
                )
            ),
            verification_fn=(
                lambda **kwargs: (
                    make_verification()
                )
            ),
        )
    )

    value = result.platform_value_add

    assert (
        value.verification_status
        == "NOT_AVAILABLE"
    )
    assert (
        value.knowledge_base_support
        is None
    )
    assert 0 <= value.health_score <= 100


def test_missing_context_recall_stays_unavailable():
    result = (
        compare_base_rag_with_platform(
            project_id="project-1",
            question="What is RAG?",
            answer="Example answer.",
            contexts=["Example context."],
            evaluation_fn=(
                lambda **kwargs: (
                    make_evaluation(
                        context_recall=None,
                    )
                )
            ),
            verification_fn=(
                lambda **kwargs: (
                    make_verification()
                )
            ),
        )
    )

    assert (
        result.core_metric_deltas[
            "context_recall"
        ]
        is None
    )
