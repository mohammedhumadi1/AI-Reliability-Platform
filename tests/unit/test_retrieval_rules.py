"""
Unit tests for retrieval_rules.py
"""

from root_cause.rules.retrieval_rules import (
    check_retrieval_failure,
    check_generation_hallucination,
)


def test_retrieval_failure_triggers_on_low_precision():
    result = check_retrieval_failure(context_precision=0.2)
    assert result is not None
    assert result["category"] == "RETRIEVAL_FAILURE"
    assert result["severity"] == "HIGH"


def test_retrieval_failure_does_not_trigger_on_good_precision():
    result = check_retrieval_failure(context_precision=0.9)
    assert result is None


def test_hallucination_triggers_on_good_retrieval_low_faithfulness():
    result = check_generation_hallucination(
        context_precision=0.92,
        faithfulness=0.35,
    )
    assert result is not None
    assert result["category"] == "GENERATION_FAILURE"
    assert result["subcategory"] == "UNSUPPORTED_CLAIM"


def test_hallucination_does_not_trigger_when_retrieval_is_bad():
    # Low precision means the issue is retrieval, not generation
    result = check_generation_hallucination(
        context_precision=0.3,
        faithfulness=0.3,
    )
    assert result is None


from root_cause.rules.retrieval_rules import check_knowledge_gap


def test_knowledge_gap_triggers_on_low_recall_good_precision():
    result = check_knowledge_gap(context_recall=0.3, context_precision=0.8)
    assert result is not None
    assert result["category"] == "KNOWLEDGE_BASE_FAILURE"


def test_knowledge_gap_does_not_trigger_on_good_recall():
    result = check_knowledge_gap(context_recall=0.9, context_precision=0.8)
    assert result is None

from root_cause.rules.retrieval_rules import check_prompt_failure


def test_prompt_failure_does_not_trigger_without_prompt_evidence():
    result = check_prompt_failure(
        context_precision=0.85,
        answer_relevancy=0.3,
        prompt_evidence=None,
    )

    assert result is None


def test_prompt_failure_triggers_with_direct_prompt_evidence():
    prompt_evidence = {
        "issue_code": "CONFLICTING_INSTRUCTIONS",
        "explanation": (
            "The prompt contains mutually conflicting "
            "instructions."
        ),
        "confidence": 0.95,
    }

    result = check_prompt_failure(
        context_precision=0.85,
        answer_relevancy=0.3,
        prompt_evidence=prompt_evidence,
    )

    assert result is not None
    assert result["category"] == "PROMPT_FAILURE"
    assert (
        result["subcategory"]
        == "CONFLICTING_INSTRUCTIONS"
    )
    assert result["confidence"] == 0.95
    assert (
        "mutually conflicting instructions"
        in result["explanation"]
    )


def test_prompt_failure_does_not_override_bad_retrieval():
    prompt_evidence = {
        "issue_code": "CONFLICTING_INSTRUCTIONS",
        "explanation": (
            "The prompt contains mutually conflicting "
            "instructions."
        ),
        "confidence": 0.95,
    }

    result = check_prompt_failure(
        context_precision=0.4,
        answer_relevancy=0.3,
        prompt_evidence=prompt_evidence,
    )

    assert result is None


from root_cause.rules.retrieval_rules import check_verified_knowledge_gap


def test_verified_knowledge_gap_triggers_when_not_supported():
    result = check_verified_knowledge_gap(
        is_supported=False,
        similarity_distance=0.95,
        explanation="No relevant documents found.",
    )
    assert result is not None
    assert result["category"] == "KNOWLEDGE_BASE_FAILURE"
    assert result["severity"] == "HIGH"


def test_verified_knowledge_gap_does_not_trigger_when_supported():
    result = check_verified_knowledge_gap(
        is_supported=True,
        similarity_distance=0.3,
        explanation="Found in document.",
    )
    assert result is None
