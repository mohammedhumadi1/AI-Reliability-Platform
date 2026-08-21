from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def calculate_binary_metrics(
    y_true: list[bool],
    y_pred: list[bool],
) -> BinaryClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if not y_true:
        raise ValueError(
            "At least one sample is required."
        )

    tp = tn = fp = fn = 0

    for truth, prediction in zip(y_true, y_pred):
        if truth and prediction:
            tp += 1
        elif not truth and not prediction:
            tn += 1
        elif not truth and prediction:
            fp += 1
        else:
            fn += 1

    total = len(y_true)

    accuracy = (tp + tn) / total

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return BinaryClassificationMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
    )
