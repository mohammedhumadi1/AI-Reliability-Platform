from evaluation.pipeline import (
    EvaluationPipelineResult,
)
from knowledge_base.verification_agent import (
    VerificationResult,
)

from benchmarks.ablation.kb_verification import (
    run_kb_verification_ablation,
)


def make_evaluation(
    *,
    context_precision=0.8,
    faithfulness=0.8,
    context_recall=0.8,
    answer_relevancy=0.8,
):
    return EvaluationPipelineResult(
        correctness_score=0.8,
        faithfulness_score=faithfulness,
        context_precision_score=context_precision,
        context_recall_score=context_recall,
        answer_relevancy_score=answer_relevancy,
        hallucination_risk=0.2,
        status="GOOD",
        explanation="Controlled evaluation.",
    )


def make_verification(
    *,
    status,
    context_alignment=None,
    answer_support=None,
):
    return VerificationResult(
        status=status,
        evidence_found=(
            status not in {
                "NOT_AVAILABLE",
                "NO_RELEVANT_EVIDENCE",
            }
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
        best_match_text=(
            "Independent company evidence."
            if status != "NOT_AVAILABLE"
            else ""
        ),
        best_match_source=(
            "company.pdf"
            if status != "NOT_AVAILABLE"
            else ""
        ),
        similarity_distance=(
            0.1
            if status != "NOT_AVAILABLE"
            else None
        ),
        question_relevance_score=(
            0.9
            if status not in {
                "NOT_AVAILABLE",
                "NO_RELEVANT_EVIDENCE",
            }
            else (
                0.0
                if status == "NO_RELEVANT_EVIDENCE"
                else None
            )
        ),
        answer_support_score=answer_support,
        context_alignment_score=context_alignment,
        numeric_contradiction=(
            status == "CONTRADICTED"
        ),
        explanation=(
            "Controlled verification result."
        ),
    )


def test_ablation_runs_core_evaluation_once():
    calls = {
        "evaluation": 0,
        "verification": 0,
    }

    evaluation = make_evaluation()

    def fake_evaluation(**kwargs):
        calls["evaluation"] += 1
        return evaluation

    def fake_verification(**kwargs):
        calls["verification"] += 1
        return make_verification(
            status="SUPPORTED",
            context_alignment=0.9,
            answer_support=0.9,
        )

    result = run_kb_verification_ablation(
        project_id="project-1",
        question="What is RAG?",
        answer="A grounded answer.",
        contexts=["Relevant context."],
        evaluation_fn=fake_evaluation,
        verification_fn=fake_verification,
    )

    assert calls["evaluation"] == 1
    assert calls["verification"] == 1
    assert result.evaluation is evaluation
    assert (
        result.without_kb.verification_status
        == "NOT_AVAILABLE"
    )
    assert (
        result.with_kb.verification_status
        == "SUPPORTED"
    )


def test_verified_evidence_can_add_generation_diagnosis():
    result = run_kb_verification_ablation(
        project_id="project-1",
        question="What is RAG?",
        answer="Unsupported answer.",
        contexts=["Relevant context."],
        evaluation_fn=(
            lambda **kwargs: make_evaluation()
        ),
        verification_fn=(
            lambda **kwargs: make_verification(
                status="UNSUPPORTED",
                context_alignment=0.8,
                answer_support=0.3,
            )
        ),
    )

    assert (
        result.without_kb.diagnosis_category
        is None
    )
    assert (
        result.with_kb.diagnosis_category
        == "GENERATION_FAILURE"
    )
    assert (
        result.with_kb.diagnosis_subcategory
        == "VERIFIED_UNSUPPORTED_ANSWER"
    )
    assert result.deltas.diagnosis_changed is True
    assert (
        result.deltas.diagnosis_subcategory_changed
        is True
    )
    assert (
        result.deltas.diagnosis_confidence_delta
        is None
    )
    assert (
        result.deltas.recommendation_count_delta
        == 3
    )
    assert (
        result.deltas.recommendations_changed
        is True
    )


def test_missing_verified_evidence_adds_kb_failure():
    result = run_kb_verification_ablation(
        project_id="project-1",
        question="What is the policy?",
        answer="Example answer.",
        contexts=["Relevant context."],
        evaluation_fn=(
            lambda **kwargs: make_evaluation()
        ),
        verification_fn=(
            lambda **kwargs: make_verification(
                status="NO_RELEVANT_EVIDENCE",
            )
        ),
    )

    assert (
        result.without_kb.diagnosis_category
        is None
    )
    assert (
        result.with_kb.diagnosis_category
        == "KNOWLEDGE_BASE_FAILURE"
    )
    assert (
        result.with_kb.diagnosis_subcategory
        == "VERIFIED_MISSING_INFORMATION"
    )
    assert result.deltas.diagnosis_changed is True


def test_unavailable_kb_preserves_proxy_only_diagnosis():
    result = run_kb_verification_ablation(
        project_id="project-1",
        question="What is RAG?",
        answer="Example answer.",
        contexts=["Weak context."],
        evaluation_fn=(
            lambda **kwargs: make_evaluation(
                context_precision=0.2,
            )
        ),
        verification_fn=(
            lambda **kwargs: make_verification(
                status="NOT_AVAILABLE",
            )
        ),
    )

    assert (
        result.without_kb.diagnosis_category
        == "RETRIEVAL_FAILURE"
    )
    assert (
        result.with_kb.diagnosis_category
        == "RETRIEVAL_FAILURE"
    )
    assert result.deltas.diagnosis_changed is False
    assert (
        result.deltas.diagnosis_subcategory_changed
        is False
    )
    assert result.deltas.health_score_delta == 0
    assert (
        result.deltas.diagnosis_confidence_delta
        == 0.0
    )
    assert (
        result.deltas.recommendation_count_delta
        == 0
    )
    assert (
        result.deltas.health_status_changed
        is False
    )
    assert (
        result.deltas.recommendations_changed
        is False
    )

def test_verified_missed_evidence_adds_retrieval_failure():
    result = run_kb_verification_ablation(
        project_id="project-1",
        question="What is the policy?",
        answer="Unsupported answer.",
        contexts=["Wrong retrieved context."],
        evaluation_fn=(
            lambda **kwargs: make_evaluation()
        ),
        verification_fn=(
            lambda **kwargs: make_verification(
                status="UNSUPPORTED",
                context_alignment=0.2,
                answer_support=0.3,
            )
        ),
    )

    assert (
        result.without_kb.diagnosis_category
        is None
    )
    assert (
        result.with_kb.diagnosis_category
        == "RETRIEVAL_FAILURE"
    )
    assert (
        result.with_kb.diagnosis_subcategory
        == "VERIFIED_MISSED_EVIDENCE"
    )
    assert (
        result.with_kb.diagnosis_confidence
        == 0.95
    )
    assert result.deltas.diagnosis_changed is True
    assert (
        result.deltas.diagnosis_subcategory_changed
        is True
    )
    assert (
        result.deltas.recommendation_count_delta
        == 3
    )
    assert (
        result.deltas.recommendations_changed
        is True
    )
