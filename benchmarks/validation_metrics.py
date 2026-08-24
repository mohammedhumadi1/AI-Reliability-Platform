from __future__ import annotations

from dataclasses import asdict, dataclass

from benchmarks.validation_schema import (
    FailureLabel,
    ValidationSample,
)


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MulticlassMetrics:
    labels: tuple[FailureLabel, ...]
    confusion_matrix: tuple[
        tuple[int, ...],
        ...,
    ]
    per_class: dict[
        FailureLabel,
        ClassMetrics,
    ]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    sample_count: int


@dataclass(frozen=True)
class ReviewerAgreement:
    reviewer_a: str
    reviewer_b: str
    sample_count: int
    observed_agreement: float
    expected_agreement: float
    cohen_kappa: float


def calculate_multiclass_metrics(
    y_true: list[FailureLabel],
    y_pred: list[FailureLabel],
    labels: tuple[
        FailureLabel,
        ...,
    ] = tuple(FailureLabel),
) -> MulticlassMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if not y_true:
        raise ValueError(
            "At least one sample is required."
        )

    if not labels:
        raise ValueError(
            "At least one label is required."
        )

    if len(labels) != len(set(labels)):
        raise ValueError(
            "labels must be unique."
        )

    allowed = set(labels)

    for value in y_true + y_pred:
        if value not in allowed:
            raise ValueError(
                f"Unknown label: {value}"
            )

    label_index = {
        label: index
        for index, label in enumerate(labels)
    }

    matrix = [
        [0 for _ in labels]
        for _ in labels
    ]

    for truth, prediction in zip(
        y_true,
        y_pred,
    ):
        matrix[
            label_index[truth]
        ][
            label_index[prediction]
        ] += 1

    per_class: dict[
        FailureLabel,
        ClassMetrics,
    ] = {}

    total = len(y_true)
    correct = 0

    for label in labels:
        index = label_index[label]

        tp = matrix[index][index]
        correct += tp

        support = sum(matrix[index])

        predicted_count = sum(
            row[index]
            for row in matrix
        )

        fp = predicted_count - tp
        fn = support - tp

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    macro_f1 = sum(
        values.f1
        for values in per_class.values()
    ) / len(labels)

    weighted_f1 = sum(
        values.f1 * values.support
        for values in per_class.values()
    ) / total

    return MulticlassMetrics(
        labels=labels,
        confusion_matrix=tuple(
            tuple(row)
            for row in matrix
        ),
        per_class=per_class,
        accuracy=correct / total,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        sample_count=total,
    )


def calculate_reviewer_agreement(
    samples: list[ValidationSample],
    reviewer_a: str,
    reviewer_b: str,
) -> ReviewerAgreement:
    reviewer_a = reviewer_a.strip()
    reviewer_b = reviewer_b.strip()

    if not reviewer_a or not reviewer_b:
        raise ValueError(
            "Reviewer IDs cannot be empty."
        )

    if reviewer_a == reviewer_b:
        raise ValueError(
            "Reviewer IDs must be different."
        )

    paired_labels: list[
        tuple[FailureLabel, FailureLabel]
    ] = []

    for sample in samples:
        annotations = {
            annotation.reviewer_id: (
                annotation.label
            )
            for annotation in sample.reviewers
        }

        if (
            reviewer_a in annotations
            and reviewer_b in annotations
        ):
            paired_labels.append(
                (
                    annotations[reviewer_a],
                    annotations[reviewer_b],
                )
            )

    if not paired_labels:
        raise ValueError(
            "No samples contain annotations "
            "from both reviewers."
        )

    sample_count = len(paired_labels)

    observed_matches = sum(
        first == second
        for first, second in paired_labels
    )

    observed_agreement = (
        observed_matches / sample_count
    )

    expected_agreement = 0.0

    for label in FailureLabel:
        count_a = sum(
            first == label
            for first, _ in paired_labels
        )

        count_b = sum(
            second == label
            for _, second in paired_labels
        )

        expected_agreement += (
            (count_a / sample_count)
            * (count_b / sample_count)
        )

    if expected_agreement == 1.0:
        cohen_kappa = (
            1.0
            if observed_agreement == 1.0
            else 0.0
        )
    else:
        cohen_kappa = (
            observed_agreement
            - expected_agreement
        ) / (
            1.0
            - expected_agreement
        )

    return ReviewerAgreement(
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        sample_count=sample_count,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        cohen_kappa=cohen_kappa,
    )
