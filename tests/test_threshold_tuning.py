import pytest

from benchmarks.tune_embedding_threshold import (
    select_best_threshold,
)


def test_threshold_tuning_finds_perfect_separation() -> None:
    threshold, metrics = (
        select_best_threshold(
            y_true=[
                True,
                True,
                False,
                False,
            ],
            scores=[
                0.90,
                0.80,
                0.20,
                0.10,
            ],
        )
    )

    assert 0.20 < threshold <= 0.80

    assert (
        metrics.balanced_accuracy
        == pytest.approx(1.0)
    )

    assert metrics.recall == pytest.approx(
        1.0
    )

    assert (
        metrics.specificity
        == pytest.approx(1.0)
    )


def test_threshold_tuning_rejects_empty_samples() -> None:
    with pytest.raises(
        ValueError,
        match="At least one sample",
    ):
        select_best_threshold(
            [],
            [],
        )
