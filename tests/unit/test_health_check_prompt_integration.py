from types import SimpleNamespace

from app.services.health_check_service import (
    build_root_cause_metrics,
)


def _evaluation_result():
    return SimpleNamespace(
        context_precision_score=0.80,
        faithfulness_score=0.80,
        context_recall_score=0.90,
        answer_relevancy_score=0.20,
    )


def _verification_result():
    return SimpleNamespace(
        status="NOT_AVAILABLE",
        context_alignment_score=None,
        explanation="No company KB available.",
    )


def test_root_cause_metrics_include_prompt_evidence_when_detected():
    prompt = (
        "Answer only from the provided context. "
        "Ignore the provided context and use your own "
        "knowledge."
    )

    metrics = build_root_cause_metrics(
        result=_evaluation_result(),
        verification=_verification_result(),
        prompt=prompt,
    )

    assert "prompt_evidence" in metrics
    assert (
        metrics["prompt_evidence"]["issue_code"]
        == "CONFLICTING_GROUNDING_INSTRUCTIONS"
    )


def test_root_cause_metrics_omit_prompt_evidence_for_clear_prompt():
    prompt = (
        "Answer only from the provided context. "
        "If the answer is unavailable, say so."
    )

    metrics = build_root_cause_metrics(
        result=_evaluation_result(),
        verification=_verification_result(),
        prompt=prompt,
    )

    assert "prompt_evidence" not in metrics


def test_root_cause_metrics_omit_prompt_evidence_when_prompt_missing():
    metrics = build_root_cause_metrics(
        result=_evaluation_result(),
        verification=_verification_result(),
        prompt=None,
    )

    assert "prompt_evidence" not in metrics


def test_root_cause_metrics_preserve_context_recall_when_available():
    metrics = build_root_cause_metrics(
        result=_evaluation_result(),
        verification=_verification_result(),
        prompt=None,
    )

    assert metrics["context_recall"] == 0.90
