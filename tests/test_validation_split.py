import pytest

from benchmarks.validation_schema import (
    FailureLabel,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)
from benchmarks.validation_split import (
    require_development_only,
    require_held_out_only,
    split_development_and_held_out,
    validate_no_split_leakage,
)


def _samples(
    per_label: int = 4,
) -> list[ValidationSample]:
    samples = []

    for label in FailureLabel:
        for index in range(per_label):
            samples.append(
                ValidationSample(
                    sample_id=(
                        f"{label.value}-"
                        f"{index}"
                    ),
                    split=(
                        ValidationSplit.DEVELOPMENT
                    ),
                    gold_label=label,
                    language=(
                        SupportedLanguage.ENGLISH
                        if index % 2 == 0
                        else SupportedLanguage.ARABIC
                    ),
                    domain=(
                        "support"
                        if index % 2 == 0
                        else "policy"
                    ),
                    question="Question?",
                    answer="Answer.",
                    contexts=("Context.",),
                )
            )

    return samples


def test_split_is_reproducible() -> None:
    samples = _samples()

    first_dev, first_test = (
        split_development_and_held_out(
            samples,
            development_fraction=0.75,
            seed=123,
        )
    )

    second_dev, second_test = (
        split_development_and_held_out(
            samples,
            development_fraction=0.75,
            seed=123,
        )
    )

    assert [
        sample.sample_id
        for sample in first_dev
    ] == [
        sample.sample_id
        for sample in second_dev
    ]

    assert [
        sample.sample_id
        for sample in first_test
    ] == [
        sample.sample_id
        for sample in second_test
    ]


def test_split_preserves_every_label_in_both_sets() -> None:
    development, held_out = (
        split_development_and_held_out(
            _samples(),
            development_fraction=0.75,
            seed=123,
        )
    )

    development_labels = {
        sample.gold_label
        for sample in development
    }

    held_out_labels = {
        sample.gold_label
        for sample in held_out
    }

    assert development_labels == set(
        FailureLabel
    )
    assert held_out_labels == set(
        FailureLabel
    )

    assert all(
        sample.split
        == ValidationSplit.DEVELOPMENT
        for sample in development
    )

    assert all(
        sample.split
        == ValidationSplit.HELD_OUT
        for sample in held_out
    )


def test_split_has_no_sample_overlap() -> None:
    development, held_out = (
        split_development_and_held_out(
            _samples(),
            seed=456,
        )
    )

    development_ids = {
        sample.sample_id
        for sample in development
    }

    held_out_ids = {
        sample.sample_id
        for sample in held_out
    }

    assert not (
        development_ids
        & held_out_ids
    )


def test_split_rejects_duplicate_sample_ids() -> None:
    samples = _samples()

    samples.append(
        samples[0]
    )

    with pytest.raises(
        ValueError,
        match="sample_id values must be unique",
    ):
        split_development_and_held_out(
            samples
        )


def test_split_requires_all_failure_labels() -> None:
    samples = [
        sample
        for sample in _samples()
        if sample.gold_label
        != FailureLabel.PROMPT_FAILURE
    ]

    with pytest.raises(
        ValueError,
        match="PROMPT_FAILURE",
    ):
        split_development_and_held_out(
            samples
        )


def test_split_requires_two_samples_per_label() -> None:
    samples = _samples(
        per_label=2
    )

    samples = [
        sample
        for sample in samples
        if not (
            sample.gold_label
            == FailureLabel.PROMPT_FAILURE
            and sample.sample_id.endswith(
                "-1"
            )
        )
    ]

    with pytest.raises(
        ValueError,
        match="PROMPT_FAILURE",
    ):
        split_development_and_held_out(
            samples
        )


def test_split_rejects_invalid_fraction() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        split_development_and_held_out(
            _samples(),
            development_fraction=1.0,
        )


def test_validate_no_split_leakage_rejects_overlap() -> None:
    sample = _samples()[0]

    development = [
        ValidationSample(
            sample_id=sample.sample_id,
            split=ValidationSplit.DEVELOPMENT,
            gold_label=sample.gold_label,
            language=sample.language,
            domain=sample.domain,
            question=sample.question,
            answer=sample.answer,
            contexts=sample.contexts,
        )
    ]

    held_out = [
        ValidationSample(
            sample_id=sample.sample_id,
            split=ValidationSplit.HELD_OUT,
            gold_label=sample.gold_label,
            language=sample.language,
            domain=sample.domain,
            question=sample.question,
            answer=sample.answer,
            contexts=sample.contexts,
        )
    ]

    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        validate_no_split_leakage(
            development,
            held_out,
        )


def test_require_development_only_rejects_held_out() -> None:
    development, held_out = (
        split_development_and_held_out(
            _samples()
        )
    )

    require_development_only(
        development
    )

    with pytest.raises(
        ValueError,
        match="development data only",
    ):
        require_development_only(
            development
            + [held_out[0]]
        )


def test_require_held_out_only_rejects_development() -> None:
    development, held_out = (
        split_development_and_held_out(
            _samples()
        )
    )

    require_held_out_only(
        held_out
    )

    with pytest.raises(
        ValueError,
        match="held-out data only",
    ):
        require_held_out_only(
            held_out
            + [development[0]]
        )
