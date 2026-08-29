"""
Unit tests for the group-aware split logic added to
benchmarks/validation_split.py, per Mohammed's review: prove that
samples sharing the same source_fact_id are never separated across
development and held-out, EVEN when they carry different gold_labels,
and that validate_no_split_leakage rejects any such overlap.
"""
from __future__ import annotations

import pytest

from benchmarks.validation_schema import (
    FailureLabel,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)
from benchmarks.validation_split import (
    split_development_and_held_out,
    validate_no_split_leakage,
)


def _make_sample(
    sample_id: str,
    label: FailureLabel,
    source_fact_id: str | None,
    split: ValidationSplit = ValidationSplit.DEVELOPMENT,
) -> ValidationSample:
    return ValidationSample(
        sample_id=sample_id,
        split=split,
        gold_label=label,
        language=SupportedLanguage.ENGLISH,
        domain="it_support",
        question=f"Question for {sample_id}?",
        answer="An answer.",
        contexts=("Some context.",),
        model_provider="groq",
        model_name="test-model",
        retriever_name="test-retriever",
        source_fact_id=source_fact_id,
    )


def _build_dataset_with_cross_label_shared_fact() -> list[ValidationSample]:
    """Build a dataset where source_fact_id 'it_support_0' is shared
    between a HEALTHY sample and a GENERATION_FAILURE sample — this is
    exactly the cross-label leakage scenario Mohammed flagged."""
    samples = []

    # A shared fact used by TWO different labels.
    samples.append(_make_sample("s-healthy-shared", FailureLabel.HEALTHY, "it_support_0"))
    samples.append(_make_sample("s-genfail-shared", FailureLabel.GENERATION_FAILURE, "it_support_0"))

    # Enough additional distinct-fact samples so every label has >= 2
    # samples and the split can place each label on both sides.
    for label in FailureLabel:
        for i in range(1, 6):
            samples.append(
                _make_sample(
                    f"s-{label.value.lower()}-{i}",
                    label,
                    f"it_support_{label.value.lower()}_{i}",
                )
            )

    return samples


def test_shared_source_fact_id_stays_on_one_side_across_labels():
    """The core requirement: a source_fact_id shared by samples with
    DIFFERENT gold_labels must still land entirely on one side."""
    samples = _build_dataset_with_cross_label_shared_fact()

    development, held_out = split_development_and_held_out(samples)

    dev_ids = {s.sample_id for s in development}
    held_ids = {s.sample_id for s in held_out}

    # Both members of the shared-fact group must be on the SAME side.
    healthy_in_dev = "s-healthy-shared" in dev_ids
    genfail_in_dev = "s-genfail-shared" in dev_ids

    assert healthy_in_dev == genfail_in_dev, (
        "Samples sharing source_fact_id='it_support_0' across "
        "different labels must land on the same side of the split."
    )

    # Sanity: no sample_id appears on both sides.
    assert dev_ids.isdisjoint(held_ids)


def test_split_result_has_no_source_fact_id_leakage():
    """After a real split run, validate_no_split_leakage must pass
    (no exception) because the split logic itself prevents leakage."""
    samples = _build_dataset_with_cross_label_shared_fact()

    development, held_out = split_development_and_held_out(samples)

    # Should not raise.
    validate_no_split_leakage(development, held_out)

    dev_facts = {s.source_fact_id for s in development if s.source_fact_id}
    held_facts = {s.source_fact_id for s in held_out if s.source_fact_id}

    assert dev_facts.isdisjoint(held_facts), (
        "No source_fact_id should appear in both development and "
        "held-out after a real split."
    )


def test_validate_no_split_leakage_rejects_manual_overlap():
    """Directly construct a development/held-out pair that manually
    violates the no-overlap rule (same source_fact_id on both sides),
    and confirm validate_no_split_leakage raises ValueError."""
    development = [
        _make_sample(
            "dev-1", FailureLabel.HEALTHY, "shared_fact_x",
            split=ValidationSplit.DEVELOPMENT,
        )
    ]
    held_out = [
        _make_sample(
            "held-1", FailureLabel.RETRIEVAL_FAILURE, "shared_fact_x",
            split=ValidationSplit.HELD_OUT,
        )
    ]

    with pytest.raises(ValueError, match="source_fact_id"):
        validate_no_split_leakage(development, held_out)


def test_validate_no_split_leakage_passes_when_facts_disjoint():
    """Sanity check: no false positives — disjoint source_fact_id sets
    must NOT raise."""
    development = [
        _make_sample(
            "dev-1", FailureLabel.HEALTHY, "fact_a",
            split=ValidationSplit.DEVELOPMENT,
        )
    ]
    held_out = [
        _make_sample(
            "held-1", FailureLabel.RETRIEVAL_FAILURE, "fact_b",
            split=ValidationSplit.HELD_OUT,
        )
    ]

    # Should not raise.
    validate_no_split_leakage(development, held_out)
