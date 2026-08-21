import pytest

from benchmarks.nli_scoring import (
    classify_score,
    strict_answer_score,
)


def test_strict_answer_score_uses_weakest_claim() -> None:
    assert strict_answer_score(
        [
            0.95,
            0.82,
            0.31,
        ]
    ) == pytest.approx(
        0.31
    )


def test_strict_answer_score_empty_is_zero() -> None:
    assert strict_answer_score(
        []
    ) == 0.0


def test_classify_score() -> None:
    assert classify_score(
        0.80,
        0.50,
    )

    assert not classify_score(
        0.40,
        0.50,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        classify_score(
            0.5,
            1.5,
        )
