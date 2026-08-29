from __future__ import annotations

import hashlib
from collections import defaultdict
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


def _group_key(sample: ValidationSample) -> str:
    """Group samples that must stay together on the same side of the
    split. Samples sharing a source_fact_id represent the same
    underlying fact/case and must never be separated across
    development and held-out — REGARDLESS of which gold_label they
    have. Samples without a source_fact_id fall back to their own
    sample_id, i.e. they form a single-sample group."""
    if sample.source_fact_id:
        return f"fact::{sample.source_fact_id}"
    return f"sample::{sample.sample_id}"


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

    # --- Cross-label, group-aware assignment -------------------------
    # Build groups ACROSS THE WHOLE DATASET (not per label), so that
    # samples sharing a source_fact_id are assigned to the same side
    # no matter which gold_label they carry.
    groups: dict[str, list[ValidationSample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)

    sorted_group_keys = sorted(
        groups.keys(),
        key=lambda key: _stable_sample_key(key, seed),
    )

    total_count = len(samples)
    raw_dev_count = round(total_count * development_fraction)
    global_target_dev_count = min(
        max(raw_dev_count, 1),
        total_count - 1,
    )

    # side_of[group_key] = "dev" or "held"
    side_of: dict[str, str] = {}
    dev_count = 0

    for group_key in sorted_group_keys:
        group_samples = groups[group_key]
        if dev_count < global_target_dev_count:
            side_of[group_key] = "dev"
            dev_count += len(group_samples)
        else:
            side_of[group_key] = "held"

    def _label_coverage() -> tuple[set, set]:
        dev_labels_ = set()
        held_labels_ = set()
        for gkey, gsamples in groups.items():
            target = dev_labels_ if side_of[gkey] == "dev" else held_labels_
            for s in gsamples:
                target.add(s.gold_label)
        return dev_labels_, held_labels_

    # --- Repair pass ---------------------------------------------------
    # If any label ended up missing from one side entirely, try to move
    # a WHOLE group (never split it) from the over-represented side to
    # the deficient one — but only if doing so doesn't remove coverage
    # of any OTHER label from the side it's leaving.
    unresolved: list[str] = []

    for label in FailureLabel:
        dev_labels, held_labels = _label_coverage()

        needs_dev = label not in dev_labels
        needs_held = label not in held_labels

        if not needs_dev and not needs_held:
            continue

        target_side = "dev" if needs_dev else "held"
        source_side = "held" if needs_dev else "dev"

        candidates = [
            gkey
            for gkey in sorted_group_keys
            if side_of[gkey] == source_side
            and any(s.gold_label == label for s in groups[gkey])
        ]

        fixed = False
        for candidate in candidates:
            original_side = side_of[candidate]
            side_of[candidate] = target_side

            dev_labels_after, held_labels_after = _label_coverage()
            still_ok = (
                len(dev_labels_after) == len(FailureLabel)
                if target_side == "dev" or source_side == "dev"
                else True
            )
            # Verify BOTH sides still have every already-satisfied
            # label covered after this move (no new gaps introduced).
            all_labels = set(FailureLabel)
            new_dev_missing = all_labels - dev_labels_after
            new_held_missing = all_labels - held_labels_after

            # Only accept if this move doesn't create OTHER new gaps
            # beyond the one we're actively trying to fix.
            other_new_gaps = (
                (new_dev_missing - {label}) if target_side != "dev" else set()
            ) | (
                (new_held_missing - {label}) if target_side != "held" else set()
            )

            if label not in (new_dev_missing | new_held_missing) and not other_new_gaps:
                fixed = True
                break
            else:
                side_of[candidate] = original_side  # revert

        if not fixed:
            unresolved.append(label.value)

    if unresolved:
        raise ValueError(
            "Group-aware split could not represent every label on "
            "both sides even after repair attempts. Labels still "
            "missing from one side: " + ", ".join(sorted(set(unresolved)))
            + ". This usually means a label's samples are all bundled "
            "into source_fact_id groups shared with other labels, "
            "leaving no movable group to fix coverage — consider more "
            "granular source_fact_id grouping."
        )

    # --- Build final development/held_out lists -------------------------
    development: list[ValidationSample] = []
    held_out: list[ValidationSample] = []

    for gkey, gsamples in groups.items():
        if side_of[gkey] == "dev":
            for sample in gsamples:
                development.append(
                    replace(sample, split=ValidationSplit.DEVELOPMENT)
                )
        else:
            for sample in gsamples:
                held_out.append(
                    replace(sample, split=ValidationSplit.HELD_OUT)
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

    development_facts = {
        sample.source_fact_id
        for sample in development
        if sample.source_fact_id
    }

    held_out_facts = {
        sample.source_fact_id
        for sample in held_out
        if sample.source_fact_id
    }

    fact_overlap = development_facts & held_out_facts

    if fact_overlap:
        raise ValueError(
            "Development and held-out sets must not share the same "
            "source_fact_id (leakage). Overlapping facts: "
            + ", ".join(sorted(fact_overlap))
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
