import pytest

from benchmarks.metrics import (
    calculate_binary_metrics,
)


def test_calculate_binary_metrics() -> None:
    metrics = calculate_binary_metrics(
        y_true=[
            True,
            True,
            False,
            False,
        ],
        y_pred=[
            True,
            False,
            True,
            False,
        ],
    )

    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.fp == 1
    assert metrics.fn == 1

    assert metrics.accuracy == pytest.approx(
        0.5
    )

    assert (
        metrics.balanced_accuracy
        == pytest.approx(0.5)
    )

    assert metrics.precision == pytest.approx(
        0.5
    )

    assert metrics.recall == pytest.approx(
        0.5
    )

    assert metrics.specificity == pytest.approx(
        0.5
    )

    assert metrics.f1 == pytest.approx(
        0.5
    )


def test_calculate_binary_metrics_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="At least one sample",
    ):
        calculate_binary_metrics(
            [],
            [],
        )


def test_calculate_binary_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_binary_metrics(
            [True],
            [
                True,
                False,
            ],
        )
