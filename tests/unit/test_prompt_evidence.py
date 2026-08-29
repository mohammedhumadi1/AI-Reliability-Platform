from root_cause.prompt_evidence import (
    analyze_prompt_evidence,
)


def test_prompt_evidence_returns_none_when_prompt_missing():
    assert analyze_prompt_evidence(None) is None


def test_prompt_evidence_returns_none_when_prompt_blank():
    assert analyze_prompt_evidence("   ") is None


def test_clear_grounded_prompt_has_no_failure_evidence():
    prompt = (
        "Answer the user's question using only the "
        "provided context. If the answer is not in the "
        "context, say that the information is unavailable."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_detects_conflicting_grounding_instructions_in_english():
    prompt = (
        "Answer only from the provided context. "
        "Ignore the provided context and use your own "
        "knowledge."
    )

    result = analyze_prompt_evidence(prompt)

    assert result is not None
    assert (
        result["issue_code"]
        == "CONFLICTING_GROUNDING_INSTRUCTIONS"
    )
    assert result["confidence"] == 0.95
    assert "conflicting" in result["explanation"].lower()


def test_detects_conflicting_grounding_instructions_in_arabic():
    prompt = (
        "\u0623\u062c\u0628 \u0641\u0642\u0637 \u0645\u0646 "
        "\u0627\u0644\u0633\u064a\u0627\u0642 "
        "\u0627\u0644\u0645\u0642\u062f\u0645. "
        "\u062a\u062c\u0627\u0647\u0644 "
        "\u0627\u0644\u0633\u064a\u0627\u0642 "
        "\u0627\u0644\u0645\u0642\u062f\u0645 "
        "\u0648\u0627\u0633\u062a\u062e\u062f\u0645 "
        "\u0645\u0639\u0631\u0641\u062a\u0643 "
        "\u0627\u0644\u062e\u0627\u0635\u0629."
    )

    result = analyze_prompt_evidence(prompt)

    assert result is not None
    assert (
        result["issue_code"]
        == "CONFLICTING_GROUNDING_INSTRUCTIONS"
    )
    assert result["confidence"] == 0.95


def test_negated_ignore_instruction_is_not_treated_as_conflict():
    prompt = (
        "Do not ignore the provided context. "
        "Answer only from the provided context."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_detects_external_knowledge_conflict():
    prompt = (
        "Do not use external knowledge. "
        "Use external knowledge when necessary."
    )

    result = analyze_prompt_evidence(prompt)

    assert result is not None
    assert (
        result["issue_code"]
        == "CONFLICTING_GROUNDING_INSTRUCTIONS"
    )


def test_english_contracted_negation_is_not_a_conflict():
    prompt = (
        "Answer only from the provided context. "
        "Don't ignore the provided context."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_arabic_negated_ignore_is_not_a_conflict():
    prompt = (
        "\u0623\u062c\u0628 \u0641\u0642\u0637 \u0645\u0646 "
        "\u0627\u0644\u0633\u064a\u0627\u0642 "
        "\u0627\u0644\u0645\u0642\u062f\u0645. "
        "\u0644\u0627 \u062a\u062a\u062c\u0627\u0647\u0644 "
        "\u0627\u0644\u0633\u064a\u0627\u0642 "
        "\u0627\u0644\u0645\u0642\u062f\u0645."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_smart_apostrophe_negation_is_not_a_conflict():
    prompt = (
        "Answer only from the provided context. "
        "Don\u2019t ignore the provided context."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_negated_own_knowledge_is_not_a_conflict():
    prompt = (
        "Answer only from the provided context. "
        "Do not use your own knowledge."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_extended_negation_is_not_a_conflict():
    prompt = (
        "Answer only from the provided context. "
        "Do not ever ignore the provided context."
    )

    assert analyze_prompt_evidence(prompt) is None


def test_never_phrase_is_not_a_conflict():
    prompt = (
        "Answer only from the provided context. "
        "Never under any circumstances use external knowledge."
    )

    assert analyze_prompt_evidence(prompt) is None
