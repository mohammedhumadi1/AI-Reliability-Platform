import pytest

from benchmarks.gold_validation import (
    validate_gold_dataset,
    validate_gold_label_sample,
)
from benchmarks.validation_schema import (
    FailureLabel,
    ReviewerAnnotation,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


def _sample(
    sample_id: str,
    gold_label: FailureLabel,
    reviewer_labels: tuple[
        FailureLabel,
        ...,
    ],
) -> ValidationSample:
    return ValidationSample(
        sample_id=sample_id,
        split=ValidationSplit.DEVELOPMENT,
        gold_label=gold_label,
        language=SupportedLanguage.ENGLISH,
        domain="support",
        question="Question?",
        answer="Answer.",
        contexts=("Context.",),
        reviewers=tuple(
            ReviewerAnnotation(
                reviewer_id=f"reviewer-{index}",
                label=label,
            )
            for index, label in enumerate(
                reviewer_labels,
                start=1,
            )
        ),
    )


def test_unanimous_reviewers_validate_gold_label() -> None:
    sample = _sample(
        "sample-1",
        FailureLabel.GENERATION_FAILURE,
        (
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ),
    )

    result = validate_gold_label_sample(
        sample
    )

    assert result.reviewer_count == 2
    assert result.unanimous is True
    assert (
        result.consensus_label
        == FailureLabel.GENERATION_FAILURE
    )
    assert result.requires_adjudication is False


def test_unanimous_reviewers_must_match_gold_label() -> None:
    sample = _sample(
        "sample-2",
        FailureLabel.RETRIEVAL_FAILURE,
        (
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match unanimous",
    ):
        validate_gold_label_sample(
            sample
        )


def test_disagreement_requires_adjudicator() -> None:
    sample = _sample(
        "sample-3",
        FailureLabel.GENERATION_FAILURE,
        (
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.RETRIEVAL_FAILURE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires adjudication",
    ):
        validate_gold_label_sample(
            sample
        )


def test_independent_adjudicator_resolves_disagreement() -> None:
    sample = _sample(
        "sample-4",
        FailureLabel.GENERATION_FAILURE,
        (
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.RETRIEVAL_FAILURE,
        ),
    )

    result = validate_gold_label_sample(
        sample,
        adjudicator_id="reviewer-3",
    )

    assert result.unanimous is False
    assert result.consensus_label is None
    assert result.requires_adjudication is True


def test_adjudicator_must_be_independent() -> None:
    sample = _sample(
        "sample-5",
        FailureLabel.GENERATION_FAILURE,
        (
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.RETRIEVAL_FAILURE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="independent",
    ):
        validate_gold_label_sample(
            sample,
            adjudicator_id="reviewer-1",
        )


def test_gold_sample_requires_two_reviewers() -> None:
    sample = _sample(
        "sample-6",
        FailureLabel.HEALTHY,
        (FailureLabel.HEALTHY,),
    )

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        validate_gold_label_sample(
            sample
        )


def test_gold_dataset_rejects_duplicate_ids() -> None:
    sample = _sample(
        "sample-7",
        FailureLabel.HEALTHY,
        (
            FailureLabel.HEALTHY,
            FailureLabel.HEALTHY,
        ),
    )

    with pytest.raises(
        ValueError,
        match="sample IDs must be unique",
    ):
        validate_gold_dataset(
            [sample, sample]
        )


def test_gold_dataset_uses_adjudicator_mapping() -> None:
    base = _sample(
        "sample-8",
        FailureLabel.PROMPT_FAILURE,
        (
            FailureLabel.PROMPT_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ),
    )

    sample = ValidationSample(
        sample_id=base.sample_id,
        split=base.split,
        gold_label=base.gold_label,
        language=base.language,
        domain=base.domain,
        question=base.question,
        answer=base.answer,
        contexts=base.contexts,
        prompt="Answer only from the supplied context.",
        reviewers=base.reviewers,
    )

    results = validate_gold_dataset(
        [sample],
        adjudicators={
            "sample-8": "reviewer-3",
        },
    )

    assert len(results) == 1
    assert results[0].requires_adjudication is True


def test_prompt_failure_requires_prompt_evidence() -> None:
    sample = _sample(
        "sample-prompt",
        FailureLabel.PROMPT_FAILURE,
        (
            FailureLabel.PROMPT_FAILURE,
            FailureLabel.PROMPT_FAILURE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="contains no prompt evidence",
    ):
        validate_gold_label_sample(
            sample
        )


def test_prompt_failure_accepts_prompt_evidence() -> None:
    base = _sample(
        "sample-prompt-evidence",
        FailureLabel.PROMPT_FAILURE,
        (
            FailureLabel.PROMPT_FAILURE,
            FailureLabel.PROMPT_FAILURE,
        ),
    )

    sample = ValidationSample(
        sample_id=base.sample_id,
        split=base.split,
        gold_label=base.gold_label,
        language=base.language,
        domain=base.domain,
        question=base.question,
        answer=base.answer,
        contexts=base.contexts,
        prompt="Answer only from supplied evidence.",
        reviewers=base.reviewers,
    )

    result = validate_gold_label_sample(
        sample
    )

    assert result.unanimous is True
    assert (
        result.consensus_label
        == FailureLabel.PROMPT_FAILURE
    )
