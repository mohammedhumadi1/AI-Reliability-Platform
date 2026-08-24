from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from benchmarks.validation_coverage import (
    ValidationCoverage,
    calculate_validation_coverage,
)
from benchmarks.validation_metrics import (
    MulticlassMetrics,
    ReviewerAgreement,
    calculate_multiclass_metrics,
    calculate_reviewer_agreement,
)
from benchmarks.validation_schema import (
    FailureLabel,
    ValidationSample,
    ValidationSplit,
)
from benchmarks.validation_split import (
    require_development_only,
    require_held_out_only,
)


@dataclass(frozen=True)
class ValidationPrediction:
    sample_id: str
    predicted_label: FailureLabel

    def __post_init__(self) -> None:
        sample_id = self.sample_id.strip()

        if not sample_id:
            raise ValueError(
                "sample_id cannot be empty."
            )

        object.__setattr__(
            self,
            "sample_id",
            sample_id,
        )


@dataclass(frozen=True)
class ValidationReport:
    split: ValidationSplit
    metrics: MulticlassMetrics
    coverage: ValidationCoverage
    reviewer_agreements: tuple[
        ReviewerAgreement,
        ...
    ]

    def to_dict(self) -> dict:
        return {
            "split": self.split.value,
            "sample_count": (
                self.metrics.sample_count
            ),
            "accuracy": (
                self.metrics.accuracy
            ),
            "macro_f1": (
                self.metrics.macro_f1
            ),
            "weighted_f1": (
                self.metrics.weighted_f1
            ),
            "labels": [
                label.value
                for label
                in self.metrics.labels
            ],
            "confusion_matrix": [
                list(row)
                for row
                in self.metrics.confusion_matrix
            ],
            "per_class": {
                label.value: {
                    "precision": values.precision,
                    "recall": values.recall,
                    "f1": values.f1,
                    "support": values.support,
                }
                for label, values
                in self.metrics.per_class.items()
            },
            "coverage": {
                "sample_count": (
                    self.coverage.sample_count
                ),
                "by_label": (
                    self.coverage.by_label
                ),
                "by_language": (
                    self.coverage.by_language
                ),
                "by_domain": (
                    self.coverage.by_domain
                ),
                "by_model": (
                    self.coverage.by_model
                ),
                "by_retriever": (
                    self.coverage.by_retriever
                ),
                "by_split": (
                    self.coverage.by_split
                ),
            },
            "reviewer_agreements": [
                {
                    "reviewer_a": (
                        agreement.reviewer_a
                    ),
                    "reviewer_b": (
                        agreement.reviewer_b
                    ),
                    "sample_count": (
                        agreement.sample_count
                    ),
                    "observed_agreement": (
                        agreement.observed_agreement
                    ),
                    "expected_agreement": (
                        agreement.expected_agreement
                    ),
                    "cohen_kappa": (
                        agreement.cohen_kappa
                    ),
                }
                for agreement
                in self.reviewer_agreements
            ],
        }


def diagnosis_to_label(
    diagnosis: dict | None,
) -> FailureLabel:
    """
    Convert the deterministic root-cause output
    into the benchmark's primary label.

    A None diagnosis means no failure rule fired,
    so it is treated as HEALTHY for this primary
    failure-classification benchmark.
    """
    if diagnosis is None:
        return FailureLabel.HEALTHY

    category = diagnosis.get(
        "category"
    )

    if not category:
        raise ValueError(
            "Diagnosis must contain a category."
        )

    try:
        return FailureLabel(
            str(category)
        )
    except ValueError as exc:
        raise ValueError(
            f"Unknown diagnosis category: "
            f"{category}"
        ) from exc


def _validate_predictions(
    samples: list[ValidationSample],
    predictions: list[
        ValidationPrediction
    ],
) -> dict[str, FailureLabel]:
    sample_ids = {
        sample.sample_id
        for sample in samples
    }

    prediction_ids = [
        prediction.sample_id
        for prediction in predictions
    ]

    if len(prediction_ids) != len(
        set(prediction_ids)
    ):
        raise ValueError(
            "Prediction sample IDs must be unique."
        )

    prediction_id_set = set(
        prediction_ids
    )

    missing = sorted(
        sample_ids
        - prediction_id_set
    )

    extra = sorted(
        prediction_id_set
        - sample_ids
    )

    if missing:
        raise ValueError(
            "Missing predictions for samples: "
            + ", ".join(missing)
        )

    if extra:
        raise ValueError(
            "Predictions contain unknown samples: "
            + ", ".join(extra)
        )

    return {
        prediction.sample_id: (
            prediction.predicted_label
        )
        for prediction in predictions
    }


def _calculate_reviewer_agreements(
    samples: list[ValidationSample],
) -> tuple[
    ReviewerAgreement,
    ...
]:
    reviewer_ids = sorted(
        {
            annotation.reviewer_id
            for sample in samples
            for annotation in sample.reviewers
        }
    )

    agreements: list[
        ReviewerAgreement
    ] = []

    for reviewer_a, reviewer_b in combinations(
        reviewer_ids,
        2,
    ):
        try:
            agreement = (
                calculate_reviewer_agreement(
                    samples,
                    reviewer_a=reviewer_a,
                    reviewer_b=reviewer_b,
                )
            )
        except ValueError as exc:
            if (
                "No samples contain annotations "
                "from both reviewers."
            ) in str(exc):
                continue

            raise

        agreements.append(
            agreement
        )

    return tuple(agreements)


def build_validation_report(
    samples: list[ValidationSample],
    predictions: list[
        ValidationPrediction
    ],
    split: ValidationSplit,
) -> ValidationReport:
    if not samples:
        raise ValueError(
            "At least one validation sample "
            "is required."
        )

    if split == ValidationSplit.DEVELOPMENT:
        require_development_only(
            samples
        )
    elif split == ValidationSplit.HELD_OUT:
        require_held_out_only(
            samples
        )
    else:
        raise ValueError(
            f"Unsupported validation split: "
            f"{split}"
        )

    prediction_map = (
        _validate_predictions(
            samples,
            predictions,
        )
    )

    y_true = [
        sample.gold_label
        for sample in samples
    ]

    y_pred = [
        prediction_map[
            sample.sample_id
        ]
        for sample in samples
    ]

    metrics = (
        calculate_multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    coverage = (
        calculate_validation_coverage(
            samples
        )
    )

    reviewer_agreements = (
        _calculate_reviewer_agreements(
            samples
        )
    )

    return ValidationReport(
        split=split,
        metrics=metrics,
        coverage=coverage,
        reviewer_agreements=(
            reviewer_agreements
        ),
    )


def save_validation_report(
    report: ValidationReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
