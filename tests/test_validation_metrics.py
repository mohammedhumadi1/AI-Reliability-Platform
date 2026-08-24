import pytest

from benchmarks.validation_metrics import (
    calculate_multiclass_metrics,
    calculate_reviewer_agreement,
)
from benchmarks.validation_schema import (
    FailureLabel,
    ReviewerAnnotation,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


def test_multiclass_metrics_perfect_predictions() -> None:
    labels = list(FailureLabel)

    metrics = calculate_multiclass_metrics(
        y_true=labels,
        y_pred=labels,
    )

    assert metrics.sample_count == 5
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.weighted_f1 == pytest.approx(1.0)

    assert metrics.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    )

    for label in FailureLabel:
        values = metrics.per_class[label]

        assert values.precision == pytest.approx(
            1.0
        )
        assert values.recall == pytest.approx(
            1.0
        )
        assert values.f1 == pytest.approx(
            1.0
        )
        assert values.support == 1


def test_multiclass_metrics_tracks_misclassification() -> None:
    metrics = calculate_multiclass_metrics(
        y_true=[
            FailureLabel.HEALTHY,
            FailureLabel.RETRIEVAL_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ],
        y_pred=[
            FailureLabel.HEALTHY,
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ],
    )

    assert metrics.sample_count == 3
    assert metrics.accuracy == pytest.approx(
        2 / 3
    )

    retrieval = metrics.per_class[
        FailureLabel.RETRIEVAL_FAILURE
    ]

    assert retrieval.precision == pytest.approx(
        0.0
    )
    assert retrieval.recall == pytest.approx(
        0.0
    )
    assert retrieval.f1 == pytest.approx(
        0.0
    )
    assert retrieval.support == 1

    generation = metrics.per_class[
        FailureLabel.GENERATION_FAILURE
    ]

    assert generation.precision == pytest.approx(
        0.5
    )
    assert generation.recall == pytest.approx(
        1.0
    )
    assert generation.f1 == pytest.approx(
        2 / 3
    )


def test_multiclass_metrics_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="At least one sample",
    ):
        calculate_multiclass_metrics(
            [],
            [],
        )


def test_multiclass_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_multiclass_metrics(
            [FailureLabel.HEALTHY],
            [],
        )


def _sample(
    sample_id: str,
    first: FailureLabel,
    second: FailureLabel,
) -> ValidationSample:
    return ValidationSample(
        sample_id=sample_id,
        split=ValidationSplit.DEVELOPMENT,
        gold_label=first,
        language=SupportedLanguage.ENGLISH,
        domain="support",
        question="Question?",
        answer="Answer.",
        contexts=("Context.",),
        reviewers=(
            ReviewerAnnotation(
                reviewer_id="reviewer-a",
                label=first,
            ),
            ReviewerAnnotation(
                reviewer_id="reviewer-b",
                label=second,
            ),
        ),
    )


def test_reviewer_agreement_perfect_match() -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            FailureLabel.HEALTHY,
        ),
        _sample(
            "two",
            FailureLabel.RETRIEVAL_FAILURE,
            FailureLabel.RETRIEVAL_FAILURE,
        ),
        _sample(
            "three",
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ),
    ]

    agreement = calculate_reviewer_agreement(
        samples,
        reviewer_a="reviewer-a",
        reviewer_b="reviewer-b",
    )

    assert agreement.sample_count == 3
    assert agreement.observed_agreement == (
        pytest.approx(1.0)
    )
    assert agreement.cohen_kappa == pytest.approx(
        1.0
    )


def test_reviewer_agreement_detects_disagreement() -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            FailureLabel.HEALTHY,
        ),
        _sample(
            "two",
            FailureLabel.RETRIEVAL_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ),
        _sample(
            "three",
            FailureLabel.GENERATION_FAILURE,
            FailureLabel.GENERATION_FAILURE,
        ),
        _sample(
            "four",
            FailureLabel.PROMPT_FAILURE,
            FailureLabel.PROMPT_FAILURE,
        ),
    ]

    agreement = calculate_reviewer_agreement(
        samples,
        reviewer_a="reviewer-a",
        reviewer_b="reviewer-b",
    )

    assert agreement.sample_count == 4
    assert agreement.observed_agreement == (
        pytest.approx(0.75)
    )
    assert -1.0 <= agreement.cohen_kappa <= 1.0


def test_reviewer_agreement_uses_only_shared_samples() -> None:
    shared = _sample(
        "shared",
        FailureLabel.HEALTHY,
        FailureLabel.HEALTHY,
    )

    single_review = ValidationSample(
        sample_id="single",
        split=ValidationSplit.DEVELOPMENT,
        gold_label=FailureLabel.HEALTHY,
        language=SupportedLanguage.ENGLISH,
        domain="support",
        question="Question?",
        answer="Answer.",
        contexts=(),
        reviewers=(
            ReviewerAnnotation(
                reviewer_id="reviewer-a",
                label=FailureLabel.HEALTHY,
            ),
        ),
    )

    agreement = calculate_reviewer_agreement(
        [shared, single_review],
        reviewer_a="reviewer-a",
        reviewer_b="reviewer-b",
    )

    assert agreement.sample_count == 1
    assert agreement.cohen_kappa == pytest.approx(
        1.0
    )


def test_reviewer_agreement_requires_shared_samples() -> None:
    sample = ValidationSample(
        sample_id="single",
        split=ValidationSplit.DEVELOPMENT,
        gold_label=FailureLabel.HEALTHY,
        language=SupportedLanguage.ENGLISH,
        domain="support",
        question="Question?",
        answer="Answer.",
        contexts=(),
        reviewers=(),
    )

    with pytest.raises(
        ValueError,
        match="both reviewers",
    ):
        calculate_reviewer_agreement(
            [sample],
            reviewer_a="reviewer-a",
            reviewer_b="reviewer-b",
        )


def test_reviewer_agreement_requires_different_reviewers() -> None:
    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        calculate_reviewer_agreement(
            [],
            reviewer_a="reviewer-a",
            reviewer_b="reviewer-a",
        )
