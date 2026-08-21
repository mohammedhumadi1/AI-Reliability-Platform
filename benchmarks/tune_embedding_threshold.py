import json
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from benchmarks.metrics import (
    BinaryClassificationMetrics,
    calculate_binary_metrics,
)


INPUT_PATH = Path(
    "benchmark_results/"
    "embedding_ragbench_validation_v1.json"
)

OUTPUT_PATH = Path(
    "benchmark_results/"
    "embedding_ragbench_validation_v1_"
    "threshold_tuning.json"
)


def select_best_threshold(
    y_true: list[bool],
    scores: list[float],
) -> tuple[
    float,
    BinaryClassificationMetrics,
]:
    if len(y_true) != len(scores):
        raise ValueError(
            "y_true and scores must have "
            "the same length."
        )

    if not y_true:
        raise ValueError(
            "At least one sample is required."
        )

    best_threshold = 0.0
    best_metrics = None
    best_key = None

    for step in range(101):
        threshold = step / 100.0

        predictions = [
            score >= threshold
            for score in scores
        ]

        metrics = (
            calculate_binary_metrics(
                y_true,
                predictions,
            )
        )

        key = (
            metrics.balanced_accuracy,
            min(
                metrics.recall,
                metrics.specificity,
            ),
            metrics.f1,
            -abs(
                threshold - 0.5
            ),
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_threshold = threshold
            best_metrics = metrics

    assert best_metrics is not None

    return (
        best_threshold,
        best_metrics,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark result not found: "
            f"{INPUT_PATH}"
        )

    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    samples = data["samples"]

    y_true = [
        bool(
            sample["truth_supported"]
        )
        for sample in samples
    ]

    scores = [
        float(
            sample["score"]
        )
        for sample in samples
    ]

    threshold, metrics = (
        select_best_threshold(
            y_true,
            scores,
        )
    )

    supported_scores = [
        score
        for truth, score
        in zip(y_true, scores)
        if truth
    ]

    unsupported_scores = [
        score
        for truth, score
        in zip(y_true, scores)
        if not truth
    ]

    roc_auc = float(
        roc_auc_score(
            y_true,
            scores,
        )
    )

    average_precision = float(
        average_precision_score(
            y_true,
            scores,
        )
    )

    per_domain = {}

    for subset in data[
        "per_domain_metrics"
    ]:
        domain_samples = [
            sample
            for sample in samples
            if sample["subset"]
            == subset
        ]

        domain_truth = [
            bool(
                sample[
                    "truth_supported"
                ]
            )
            for sample
            in domain_samples
        ]

        domain_predictions = [
            float(
                sample["score"]
            ) >= threshold
            for sample
            in domain_samples
        ]

        domain_metrics = (
            calculate_binary_metrics(
                domain_truth,
                domain_predictions,
            )
        )

        per_domain[subset] = (
            domain_metrics.to_dict()
        )

    payload = {
        "benchmark_version": (
            data["benchmark_version"]
        ),
        "model": data["model"],
        "sample_count": len(samples),
        "original_threshold": (
            data["threshold"]
        ),
        "best_threshold": threshold,
        "roc_auc": roc_auc,
        "average_precision": (
            average_precision
        ),
        "score_summary": {
            "supported_mean": (
                sum(supported_scores)
                / len(supported_scores)
            ),
            "unsupported_mean": (
                sum(unsupported_scores)
                / len(
                    unsupported_scores
                )
            ),
        },
        "best_metrics": (
            metrics.to_dict()
        ),
        "per_domain_metrics": (
            per_domain
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print(
        "EMBEDDING FAITHFULNESS "
        "THRESHOLD CALIBRATION"
    )
    print("=" * 76)

    print(
        "Samples:             ",
        len(samples),
    )

    print(
        "Original threshold:  ",
        f"{data['threshold']:.2f}",
    )

    print(
        "Best threshold:      ",
        f"{threshold:.2f}",
    )

    print("-" * 76)

    print(
        "ROC-AUC:             ",
        f"{roc_auc:.4f}",
    )

    print(
        "Average Precision:   ",
        f"{average_precision:.4f}",
    )

    print(
        "Supported mean:      ",
        f"{payload['score_summary']['supported_mean']:.4f}",
    )

    print(
        "Unsupported mean:    ",
        f"{payload['score_summary']['unsupported_mean']:.4f}",
    )

    print("-" * 76)

    print(
        "TP:                  ",
        metrics.tp,
    )

    print(
        "TN:                  ",
        metrics.tn,
    )

    print(
        "FP:                  ",
        metrics.fp,
    )

    print(
        "FN:                  ",
        metrics.fn,
    )

    print("-" * 76)

    print(
        "Accuracy:            ",
        f"{metrics.accuracy:.4f}",
    )

    print(
        "Balanced Accuracy:   ",
        f"{metrics.balanced_accuracy:.4f}",
    )

    print(
        "Precision:           ",
        f"{metrics.precision:.4f}",
    )

    print(
        "Recall:              ",
        f"{metrics.recall:.4f}",
    )

    print(
        "Specificity:         ",
        f"{metrics.specificity:.4f}",
    )

    print(
        "F1:                  ",
        f"{metrics.f1:.4f}",
    )

    print()
    print("=" * 76)
    print(
        "PER-DOMAIN @ BEST THRESHOLD"
    )
    print("=" * 76)

    for subset, values in (
        per_domain.items()
    ):
        print(
            f"{subset:<14}"
            f" BA="
            f"{values['balanced_accuracy']:.4f}"
            f" Recall="
            f"{values['recall']:.4f}"
            f" Spec="
            f"{values['specificity']:.4f}"
            f" F1="
            f"{values['f1']:.4f}"
        )

    print("-" * 76)

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
