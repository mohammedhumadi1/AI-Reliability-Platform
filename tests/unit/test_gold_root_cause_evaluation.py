from pathlib import Path

import pytest

from benchmarks.run_gold_root_cause_evaluation import (
    build_breakdown,
    load_split_samples,
)
from benchmarks.validation_runner import (
    ValidationPrediction,
)
from benchmarks.validation_schema import (
    FailureLabel,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


def _sample(
    sample_id: str,
    label: FailureLabel,
    language: SupportedLanguage,
    domain: str,
) -> ValidationSample:
    return ValidationSample(
        sample_id=sample_id,
        split=(
            ValidationSplit.DEVELOPMENT
        ),
        gold_label=label,
        language=language,
        domain=domain,
        question="Question",
        answer="Answer",
        contexts=("Context",),
    )


def test_held_out_requires_explicit_confirmation(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="--confirm-held-out",
    ):
        load_split_samples(
            split=(
                ValidationSplit.HELD_OUT
            ),
            confirm_held_out=False,
            dataset_dir=tmp_path,
        )


def test_breakdown_reports_group_accuracy() -> None:
    samples = [
        _sample(
            sample_id="one",
            label=FailureLabel.HEALTHY,
            language=(
                SupportedLanguage.ENGLISH
            ),
            domain="it_support",
        ),
        _sample(
            sample_id="two",
            label=(
                FailureLabel
                .GENERATION_FAILURE
            ),
            language=(
                SupportedLanguage.ENGLISH
            ),
            domain="it_support",
        ),
    ]

    predictions = [
        ValidationPrediction(
            sample_id="one",
            predicted_label=(
                FailureLabel.HEALTHY
            ),
        ),
        ValidationPrediction(
            sample_id="two",
            predicted_label=(
                FailureLabel.HEALTHY
            ),
        ),
    ]

    breakdown = build_breakdown(
        samples=samples,
        predictions=predictions,
        attribute="language",
    )

    assert breakdown["en"][
        "sample_count"
    ] == 2

    assert breakdown["en"][
        "correct"
    ] == 1

    assert breakdown["en"][
        "mistakes"
    ] == 1

    assert breakdown["en"][
        "accuracy"
    ] == 0.5
