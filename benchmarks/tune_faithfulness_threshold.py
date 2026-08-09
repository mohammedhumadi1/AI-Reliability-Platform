from benchmarks.ragbench_loader import load_ragbench_samples
from evaluation.evaluators.faithfulness import FaithfulnessEvaluator


SAMPLE_LIMIT = 100


def calculate_metrics(
    scores,
    labels,
    threshold,
):
    tp = tn = fp = fn = 0

    for score, label in zip(scores, labels):
        prediction = score >= threshold

        if prediction and label:
            tp += 1
        elif not prediction and not label:
            tn += 1
        elif prediction and not label:
            fp += 1
        else:
            fn += 1

    accuracy = (
        (tp + tn) / len(labels)
        if labels
        else 0.0
    )

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
        (recall + specificity) / 2
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


def main():
    print(
        f"Loading {SAMPLE_LIMIT} validation samples..."
    )

    samples = load_ragbench_samples(
        split="validation",
        limit=SAMPLE_LIMIT,
    )

    evaluator = FaithfulnessEvaluator()

    scores = []
    labels = []

    for i, sample in enumerate(
        samples,
        start=1,
    ):
        result = evaluator.evaluate(
            answer=sample["response"],
            contexts=sample["contexts"],
        )

        scores.append(result.score)
        labels.append(
            bool(sample["adherence_score"])
        )

        print(
            f"{i:03d} | "
            f"GT={labels[-1]} | "
            f"Score={result.score:.4f}"
        )

    supported = sum(labels)
    unsupported = len(labels) - supported

    print("=" * 80)
    print(f"Supported:   {supported}")
    print(f"Unsupported: {unsupported}")
    print("=" * 80)

    best_threshold = None
    best_metrics = None

    for value in range(40, 96):
        threshold = value / 100

        metrics = calculate_metrics(
            scores,
            labels,
            threshold,
        )

        if (
            best_metrics is None
            or metrics["balanced_accuracy"]
            > best_metrics["balanced_accuracy"]
            or (
                metrics["balanced_accuracy"]
                == best_metrics["balanced_accuracy"]
                and metrics["f1"]
                > best_metrics["f1"]
            )
        ):
            best_threshold = threshold
            best_metrics = metrics

    print("BEST VALIDATION THRESHOLD")
    print("=" * 80)

    print(
        f"Threshold:         "
        f"{best_threshold:.2f}"
    )
    print(
        f"Accuracy:          "
        f"{best_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{best_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Precision:         "
        f"{best_metrics['precision']:.4f}"
    )
    print(
        f"Recall:            "
        f"{best_metrics['recall']:.4f}"
    )
    print(
        f"Specificity:       "
        f"{best_metrics['specificity']:.4f}"
    )
    print(
        f"F1:                "
        f"{best_metrics['f1']:.4f}"
    )

    print("-" * 80)

    print(f"TP: {best_metrics['tp']}")
    print(f"TN: {best_metrics['tn']}")
    print(f"FP: {best_metrics['fp']}")
    print(f"FN: {best_metrics['fn']}")


if __name__ == "__main__":
    main()