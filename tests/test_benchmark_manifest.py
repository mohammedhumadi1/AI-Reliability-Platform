import pytest

from benchmarks.build_validation_manifest import (
    select_balanced_indices,
)


def test_balanced_selection_is_reproducible() -> None:
    labels = (
        [True] * 10
        + [False] * 10
    )

    first = select_balanced_indices(
        labels=labels,
        samples_per_class=4,
        seed=123,
    )

    second = select_balanced_indices(
        labels=labels,
        samples_per_class=4,
        seed=123,
    )

    assert first == second
    assert len(first) == 8

    selected_labels = [
        labels[index]
        for index in first
    ]

    assert sum(selected_labels) == 4
    assert (
        len(selected_labels)
        - sum(selected_labels)
        == 4
    )


def test_balanced_selection_rejects_insufficient_class() -> None:
    labels = [
        True,
        True,
        True,
        False,
    ]

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        select_balanced_indices(
            labels=labels,
            samples_per_class=2,
            seed=123,
        )
