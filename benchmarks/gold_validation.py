from __future__ import annotations

from dataclasses import dataclass

from benchmarks.validation_schema import (
    FailureLabel,
    ValidationSample,
)


@dataclass(frozen=True)
class GoldLabelValidationResult:
    sample_id: str
    reviewer_count: int
    unanimous: bool
    consensus_label: FailureLabel | None
    requires_adjudication: bool


def validate_gold_label_sample(
    sample: ValidationSample,
    adjudicator_id: str | None = None,
    minimum_reviewers: int = 2,
) -> GoldLabelValidationResult:
    if minimum_reviewers < 2:
        raise ValueError(
            "minimum_reviewers must be at least 2."
        )

    if (
        sample.gold_label
        == FailureLabel.PROMPT_FAILURE
        and not sample.prompt
    ):
        raise ValueError(
            f"Sample {sample.sample_id} is labeled "
            "PROMPT_FAILURE but contains no prompt "
            "evidence."
        )

    reviewer_count = len(sample.reviewers)

    if reviewer_count < minimum_reviewers:
        raise ValueError(
            f"Sample {sample.sample_id} requires at least "
            f"{minimum_reviewers} independent reviewers."
        )

    reviewer_ids = {
        annotation.reviewer_id
        for annotation in sample.reviewers
    }

    if len(reviewer_ids) != reviewer_count:
        raise ValueError(
            "Reviewer IDs must be unique."
        )

    labels = {
        annotation.label
        for annotation in sample.reviewers
    }

    unanimous = len(labels) == 1

    consensus_label = (
        next(iter(labels))
        if unanimous
        else None
    )

    if unanimous:
        if sample.gold_label != consensus_label:
            raise ValueError(
                f"Sample {sample.sample_id} gold label "
                "does not match unanimous reviewer "
                "agreement."
            )

        return GoldLabelValidationResult(
            sample_id=sample.sample_id,
            reviewer_count=reviewer_count,
            unanimous=True,
            consensus_label=consensus_label,
            requires_adjudication=False,
        )

    cleaned_adjudicator = (
        adjudicator_id.strip()
        if adjudicator_id
        else ""
    )

    if not cleaned_adjudicator:
        raise ValueError(
            f"Sample {sample.sample_id} has reviewer "
            "disagreement and requires adjudication."
        )

    if cleaned_adjudicator in reviewer_ids:
        raise ValueError(
            "The adjudicator must be independent from "
            "the original reviewers."
        )

    return GoldLabelValidationResult(
        sample_id=sample.sample_id,
        reviewer_count=reviewer_count,
        unanimous=False,
        consensus_label=None,
        requires_adjudication=True,
    )


def validate_gold_dataset(
    samples: list[ValidationSample],
    adjudicators: dict[str, str] | None = None,
    minimum_reviewers: int = 2,
) -> list[GoldLabelValidationResult]:
    if not samples:
        raise ValueError(
            "At least one gold sample is required."
        )

    sample_ids = [
        sample.sample_id
        for sample in samples
    ]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Gold dataset sample IDs must be unique."
        )

    adjudicators = adjudicators or {}

    return [
        validate_gold_label_sample(
            sample,
            adjudicator_id=adjudicators.get(
                sample.sample_id
            ),
            minimum_reviewers=minimum_reviewers,
        )
        for sample in samples
    ]
