import pytest

from benchmarks.validation_schema import (
    Adjudication,
    FailureLabel,
    ReviewerAnnotation,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


def test_validation_sample_supports_gold_metadata() -> None:
    sample = ValidationSample(
        sample_id="sample-001",
        split=ValidationSplit.DEVELOPMENT,
        gold_label=(
            FailureLabel.GENERATION_FAILURE
        ),
        language=SupportedLanguage.ENGLISH,
        domain="customer-support",
        question="What is the refund period?",
        answer="The refund period is 30 days.",
        contexts=(
            "Refunds are available within 14 days.",
        ),
        reference_answer=(
            "The refund period is 14 days."
        ),
        prompt=(
            "Answer only from the supplied context."
        ),
        reviewers=(
            ReviewerAnnotation(
                reviewer_id="reviewer-a",
                label=(
                    FailureLabel.GENERATION_FAILURE
                ),
            ),
            ReviewerAnnotation(
                reviewer_id="reviewer-b",
                label=(
                    FailureLabel.GENERATION_FAILURE
                ),
            ),
        ),
    )

    assert sample.sample_id == "sample-001"
    assert (
        sample.split
        == ValidationSplit.DEVELOPMENT
    )
    assert (
        sample.gold_label
        == FailureLabel.GENERATION_FAILURE
    )
    assert sample.language == SupportedLanguage.ENGLISH
    assert len(sample.reviewers) == 2


def test_validation_sample_cleans_optional_text() -> None:
    sample = ValidationSample(
        sample_id=" sample-002 ",
        split=ValidationSplit.HELD_OUT,
        gold_label=FailureLabel.HEALTHY,
        language=SupportedLanguage.ARABIC,
        domain=" policies ",
        question=" \u0645\u0627 \u0645\u062f\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629\u061f ",
        answer=" \u0645\u062f\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629 21 \u064a\u0648\u0645\u064b\u0627. ",
        contexts=(
            " ",
            " \u0645\u062f\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629 \u0627\u0644\u0633\u0646\u0648\u064a\u0629 21 \u064a\u0648\u0645\u064b\u0627. ",
        ),
        reference_answer=" ",
        prompt=" ",
    )

    assert sample.sample_id == "sample-002"
    assert sample.domain == "policies"
    assert sample.question == "\u0645\u0627 \u0645\u062f\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629\u061f"
    assert sample.answer == "\u0645\u062f\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629 21 \u064a\u0648\u0645\u064b\u0627."
    assert sample.contexts == (
        "\u0645\u062f\u0629 \u0627\u0644\u0625\u062c\u0627\u0632\u0629 \u0627\u0644\u0633\u0646\u0648\u064a\u0629 21 \u064a\u0648\u0645\u064b\u0627.",
    )
    assert sample.reference_answer is None
    assert sample.prompt is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sample_id", ""),
        ("domain", " "),
        ("question", ""),
        ("answer", "   "),
    ],
)
def test_validation_sample_rejects_empty_required_text(
    field_name: str,
    value: str,
) -> None:
    kwargs = {
        "sample_id": "sample-003",
        "split": ValidationSplit.DEVELOPMENT,
        "gold_label": FailureLabel.HEALTHY,
        "language": SupportedLanguage.ENGLISH,
        "domain": "support",
        "question": "Question?",
        "answer": "Answer.",
        "contexts": (),
    }

    kwargs[field_name] = value

    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be empty",
    ):
        ValidationSample(**kwargs)


def test_validation_sample_rejects_duplicate_reviewers() -> None:
    reviewer = ReviewerAnnotation(
        reviewer_id="reviewer-a",
        label=FailureLabel.HEALTHY,
    )

    with pytest.raises(
        ValueError,
        match="Reviewer IDs must be unique",
    ):
        ValidationSample(
            sample_id="sample-004",
            split=ValidationSplit.DEVELOPMENT,
            gold_label=FailureLabel.HEALTHY,
            language=SupportedLanguage.ENGLISH,
            domain="support",
            question="Question?",
            answer="Answer.",
            contexts=(),
            reviewers=(
                reviewer,
                reviewer,
            ),
        )


def test_reviewer_annotation_rejects_empty_id() -> None:
    with pytest.raises(
        ValueError,
        match="reviewer_id cannot be empty",
    ):
        ReviewerAnnotation(
            reviewer_id=" ",
            label=FailureLabel.HEALTHY,
        )



def test_adjudication_cleans_metadata() -> None:
    adjudication = Adjudication(
        adjudicator_id=" reviewer-c ",
        label=FailureLabel.GENERATION_FAILURE,
        notes=" final decision ",
    )

    assert adjudication.adjudicator_id == "reviewer-c"
    assert (
        adjudication.label
        == FailureLabel.GENERATION_FAILURE
    )
    assert adjudication.notes == "final decision"


def test_adjudication_rejects_empty_id() -> None:
    with pytest.raises(
        ValueError,
        match="adjudicator_id cannot be empty",
    ):
        Adjudication(
            adjudicator_id=" ",
            label=FailureLabel.HEALTHY,
        )
