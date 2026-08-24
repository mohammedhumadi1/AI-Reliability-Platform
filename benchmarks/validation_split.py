from __future__ import annotations

import hashlib
from dataclasses import replace

from benchmarks.validation_schema import (
    FailureLabel,
    ValidationSample,
    ValidationSplit,
)


DEFAULT_DEVELOPMENT_FRACTION = 0.70
DEFAULT_SPLIT_SEED = 20260824


def _stable_sample_key(
    sample_id: str,
    seed: int,
) -> str:
    payload = (
        f"{seed}:{sample_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def _validate_unique_sample_ids(
    samples: list[ValidationSample],
) -> None:
    sample_ids = [
        sample.sample_id
        for sample in samples
    ]

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError(
            "sample_id values must be unique."
        )


def split_development_and_held_out(
    samples: list[ValidationSample],
    development_fraction: float = (
        DEFAULT_DEVELOPMENT_FRACTION
    ),
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[
    list[ValidationSample],
    list[ValidationSample],
]:
    if not samples:
        raise ValueError(
            "At least one sample is required."
        )

    if not 0.0 < development_fraction < 1.0:
        raise ValueError(
            "development_fraction must be "
            "between 0 and 1."
        )

    _validate_unique_sample_ids(samples)

    by_label: dict[
        FailureLabel,
        list[ValidationSample],
    ] = {
        label: []
        for label in FailureLabel
    }

    for sample in samples:
        by_label[
            sample.gold_label
        ].append(sample)

    missing_labels = [
        label.value
        for label, label_samples
        in by_label.items()
        if not label_samples
    ]

    if missing_labels:
        raise ValueError(
            "Every failure label must be "
            "represented before splitting. "
            "Missing: "
            + ", ".join(missing_labels)
        )

    too_small = [
        label.value
        for label, label_samples
        in by_label.items()
        if len(label_samples) < 2
    ]

    if too_small:
        raise ValueError(
            "Each failure label needs at least "
            "two samples so both splits contain "
            "that label. Too small: "
            + ", ".join(too_small)
        )

    development: list[
        ValidationSample
    ] = []

    held_out: list[
        ValidationSample
    ] = []

    for label in FailureLabel:
        label_samples = sorted(
            by_label[label],
            key=lambda sample: (
                _stable_sample_key(
                    sample.sample_id,
                    seed,
                )
            ),
        )

        raw_dev_count = round(
            len(label_samples)
            * development_fraction
        )

        development_count = min(
            max(raw_dev_count, 1),
            len(label_samples) - 1,
        )

        for sample in label_samples[
            :development_count
        ]:
            development.append(
                replace(
                    sample,
                    split=(
                        ValidationSplit.DEVELOPMENT
                    ),
                )
            )

        for sample in label_samples[
            development_count:
        ]:
            held_out.append(
                replace(
                    sample,
                    split=(
                        ValidationSplit.HELD_OUT
                    ),
                )
            )

    development.sort(
        key=lambda sample: sample.sample_id
    )

    held_out.sort(
        key=lambda sample: sample.sample_id
    )

    validate_no_split_leakage(
        development,
        held_out,
    )

    return development, held_out


def validate_no_split_leakage(
    development: list[ValidationSample],
    held_out: list[ValidationSample],
) -> None:
    development_ids = {
        sample.sample_id
        for sample in development
    }

    held_out_ids = {
        sample.sample_id
        for sample in held_out
    }

    overlap = (
        development_ids
        & held_out_ids
    )

    if overlap:
        raise ValueError(
            "Development and held-out sets "
            "must not overlap. Duplicate IDs: "
            + ", ".join(sorted(overlap))
        )

    if any(
        sample.split
        != ValidationSplit.DEVELOPMENT
        for sample in development
    ):
        raise ValueError(
            "Development samples must use "
            "the development split."
        )

    if any(
        sample.split
        != ValidationSplit.HELD_OUT
        for sample in held_out
    ):
        raise ValueError(
            "Held-out samples must use "
            "the held_out split."
        )


def require_development_only(
    samples: list[ValidationSample],
) -> None:
    if not samples:
        raise ValueError(
            "At least one development sample "
            "is required."
        )

    invalid = [
        sample.sample_id
        for sample in samples
        if sample.split
        != ValidationSplit.DEVELOPMENT
    ]

    if invalid:
        raise ValueError(
            "Threshold tuning and rule tuning "
            "are allowed on development data "
            "only. Held-out samples detected: "
            + ", ".join(invalid)
        )


def require_held_out_only(
    samples: list[ValidationSample],
) -> None:
    if not samples:
        raise ValueError(
            "At least one held-out sample "
            "is required."
        )

    invalid = [
        sample.sample_id
        for sample in samples
        if sample.split
        != ValidationSplit.HELD_OUT
    ]

    if invalid:
        raise ValueError(
            "Final evaluation must use "
            "held-out data only. Invalid "
            "samples: "
            + ", ".join(invalid)
        )
