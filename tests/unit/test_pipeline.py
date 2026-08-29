"""
Unit tests for root_cause/rules/pipeline.py
"""

from root_cause.rules.pipeline import (
    run_rules_pipeline,
)


def test_pipeline_detects_retrieval_failure() -> None:
    metrics = {
        "context_precision": 0.2,
        "faithfulness": 0.8,
    }

    result = run_rules_pipeline(metrics)

    assert result is not None
    assert (
        result["category"]
        == "RETRIEVAL_FAILURE"
    )


def test_pipeline_detects_hallucination_when_retrieval_is_good() -> None:
    metrics = {
        "context_precision": 0.92,
        "faithfulness": 0.35,
    }

    result = run_rules_pipeline(metrics)

    assert result is not None
    assert (
        result["category"]
        == "GENERATION_FAILURE"
    )


def test_pipeline_detects_knowledge_base_gap() -> None:
    metrics = {
        "context_precision": 0.80,
        "faithfulness": 0.80,
        "context_recall": 0.20,
        "answer_relevancy": 0.90,
    }

    result = run_rules_pipeline(metrics)

    assert result is not None
    assert (
        result["category"]
        == "KNOWLEDGE_BASE_FAILURE"
    )


def test_pipeline_does_not_infer_prompt_failure_without_evidence() -> None:
    metrics = {
        "context_precision": 0.80,
        "faithfulness": 0.80,
        "context_recall": 0.90,
        "answer_relevancy": 0.20,
    }

    result = run_rules_pipeline(metrics)

    assert result is None


def test_pipeline_detects_prompt_failure_with_direct_evidence() -> None:
    metrics = {
        "context_precision": 0.80,
        "faithfulness": 0.80,
        "context_recall": 0.90,
        "answer_relevancy": 0.20,
        "prompt_evidence": {
            "issue_code": "CONFLICTING_INSTRUCTIONS",
            "explanation": (
                "The prompt contains mutually conflicting "
                "instructions."
            ),
            "confidence": 0.95,
        },
    }

    result = run_rules_pipeline(metrics)

    assert result is not None
    assert (
        result["category"]
        == "PROMPT_FAILURE"
    )
    assert (
        result["subcategory"]
        == "CONFLICTING_INSTRUCTIONS"
    )


def test_pipeline_returns_none_when_everything_is_healthy() -> None:
    metrics = {
        "context_precision": 0.9,
        "faithfulness": 0.9,
        "context_recall": 0.9,
        "answer_relevancy": 0.9,
    }

    result = run_rules_pipeline(metrics)

    assert result is None


def test_pipeline_returns_none_when_context_precision_missing() -> None:
    metrics = {
        "faithfulness": 0.9,
    }

    result = run_rules_pipeline(metrics)

    assert result is None


def test_pipeline_prioritizes_retrieval_over_hallucination() -> None:
    metrics = {
        "context_precision": 0.2,
        "faithfulness": 0.2,
        "context_recall": 0.2,
        "answer_relevancy": 0.2,
    }

    result = run_rules_pipeline(metrics)

    assert (
        result["category"]
        == "RETRIEVAL_FAILURE"
    )
