import pytest

from benchmarks.ragbench_loader import (
    select_balanced_indices,
)


def test_select_balanced_indices_is_deterministic() -> None:
    labels = [
        True,
        False,
        True,
        False,
        True,
        False,
    ]

    assert select_balanced_indices(
        labels,
        limit=4,
    ) == [
        0,
        1,
        2,
        3,
    ]


def test_select_balanced_indices_requires_both_classes() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        select_balanced_indices(
            [
                True,
                True,
                True,
            ],
            limit=2,
        )
