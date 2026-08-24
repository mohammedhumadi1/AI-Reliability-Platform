import json

import pytest

from benchmarks.validation_runner import (
    ValidationPrediction,
    build_validation_report,
    diagnosis_to_label,
    save_validation_report,
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
    label: FailureLabel,
    split: ValidationSplit,
) -> ValidationSample:
    return ValidationSample(
        sample_id=sample_id,
        split=split,
        gold_label=label,
        language=(
            SupportedLanguage.ENGLISH
        ),
        domain="support",
        question="Question?",
        answer="Answer.",
        contexts=("Context.",),
        model_provider="provider",
        model_name="model",
        retriever_name="retriever",
        reviewers=(
            ReviewerAnnotation(
                reviewer_id="reviewer-a",
                label=label,
            ),
            ReviewerAnnotation(
                reviewer_id="reviewer-b",
                label=label,
            ),
        ),
    )


def test_diagnosis_none_maps_to_healthy() -> None:
    assert (
        diagnosis_to_label(None)
        == FailureLabel.HEALTHY
    )


@pytest.mark.parametrize(
    "label",
    [
        FailureLabel.RETRIEVAL_FAILURE,
        FailureLabel.GENERATION_FAILURE,
        FailureLabel.KNOWLEDGE_BASE_FAILURE,
        FailureLabel.PROMPT_FAILURE,
    ],
)
def test_diagnosis_category_maps_to_label(
    label: FailureLabel,
) -> None:
    assert (
        diagnosis_to_label(
            {
                "category": label.value,
            }
        )
        == label
    )


def test_diagnosis_rejects_unknown_category() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown diagnosis category",
    ):
        diagnosis_to_label(
            {
                "category": "UNKNOWN",
            }
        )


def test_build_held_out_report() -> None:
    samples = [
        _sample(
            "healthy",
            FailureLabel.HEALTHY,
            ValidationSplit.HELD_OUT,
        ),
        _sample(
            "retrieval",
            FailureLabel.RETRIEVAL_FAILURE,
            ValidationSplit.HELD_OUT,
        ),
        _sample(
            "generation",
            FailureLabel.GENERATION_FAILURE,
            ValidationSplit.HELD_OUT,
        ),
        _sample(
            "kb",
            FailureLabel.KNOWLEDGE_BASE_FAILURE,
            ValidationSplit.HELD_OUT,
        ),
        _sample(
            "prompt",
            FailureLabel.PROMPT_FAILURE,
            ValidationSplit.HELD_OUT,
        ),
    ]

    predictions = [
        ValidationPrediction(
            sample_id=sample.sample_id,
            predicted_label=sample.gold_label,
        )
        for sample in samples
    ]

    report = build_validation_report(
        samples=samples,
        predictions=predictions,
        split=ValidationSplit.HELD_OUT,
    )

    assert report.split == (
        ValidationSplit.HELD_OUT
    )
    assert report.metrics.sample_count == 5
    assert report.metrics.accuracy == (
        pytest.approx(1.0)
    )
    assert report.metrics.macro_f1 == (
        pytest.approx(1.0)
    )

    assert len(
        report.reviewer_agreements
    ) == 1

    assert (
        report.reviewer_agreements[
            0
        ].cohen_kappa
        == pytest.approx(1.0)
    )


def test_report_tracks_misclassification() -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            ValidationSplit.DEVELOPMENT,
        ),
        _sample(
            "two",
            FailureLabel.RETRIEVAL_FAILURE,
            ValidationSplit.DEVELOPMENT,
        ),
    ]

    predictions = [
        ValidationPrediction(
            sample_id="one",
            predicted_label=FailureLabel.HEALTHY,
        ),
        ValidationPrediction(
            sample_id="two",
            predicted_label=(
                FailureLabel.GENERATION_FAILURE
            ),
        ),
    ]

    report = build_validation_report(
        samples=samples,
        predictions=predictions,
        split=ValidationSplit.DEVELOPMENT,
    )

    assert report.metrics.accuracy == (
        pytest.approx(0.5)
    )


def test_report_rejects_missing_prediction() -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            ValidationSplit.HELD_OUT,
        ),
        _sample(
            "two",
            FailureLabel.RETRIEVAL_FAILURE,
            ValidationSplit.HELD_OUT,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Missing predictions",
    ):
        build_validation_report(
            samples=samples,
            predictions=[
                ValidationPrediction(
                    sample_id="one",
                    predicted_label=(
                        FailureLabel.HEALTHY
                    ),
                )
            ],
            split=ValidationSplit.HELD_OUT,
        )


def test_report_rejects_unknown_prediction() -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            ValidationSplit.HELD_OUT,
        )
    ]

    with pytest.raises(
        ValueError,
        match="unknown samples",
    ):
        build_validation_report(
            samples=samples,
            predictions=[
                ValidationPrediction(
                    sample_id="one",
                    predicted_label=(
                        FailureLabel.HEALTHY
                    ),
                ),
                ValidationPrediction(
                    sample_id="extra",
                    predicted_label=(
                        FailureLabel.HEALTHY
                    ),
                ),
            ],
            split=ValidationSplit.HELD_OUT,
        )


def test_report_rejects_wrong_split() -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            ValidationSplit.DEVELOPMENT,
        )
    ]

    with pytest.raises(
        ValueError,
        match="held-out data only",
    ):
        build_validation_report(
            samples=samples,
            predictions=[
                ValidationPrediction(
                    sample_id="one",
                    predicted_label=(
                        FailureLabel.HEALTHY
                    ),
                )
            ],
            split=ValidationSplit.HELD_OUT,
        )


def test_save_validation_report(
    tmp_path,
) -> None:
    samples = [
        _sample(
            "one",
            FailureLabel.HEALTHY,
            ValidationSplit.HELD_OUT,
        )
    ]

    report = build_validation_report(
        samples=samples,
        predictions=[
            ValidationPrediction(
                sample_id="one",
                predicted_label=(
                    FailureLabel.HEALTHY
                ),
            )
        ],
        split=ValidationSplit.HELD_OUT,
    )

    output_path = (
        tmp_path
        / "validation-report.json"
    )

    save_validation_report(
        report,
        output_path,
    )

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["split"] == "held_out"
    assert payload["sample_count"] == 1
    assert payload["accuracy"] == (
        pytest.approx(1.0)
    )
    assert (
        payload["coverage"][
            "by_label"
        ]["HEALTHY"]
        == 1
    )
