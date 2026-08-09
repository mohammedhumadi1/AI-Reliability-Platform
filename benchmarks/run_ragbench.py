from benchmarks.ragbench_loader import load_ragbench_samples
from evaluation.pipeline import run_evaluation


THRESHOLD = 0.75


def classify_supported(
    faithfulness_score: float,
) -> bool:
    return faithfulness_score >= THRESHOLD


def main():
    samples = load_ragbench_samples(limit=20)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print(
        f"Faithfulness threshold: {THRESHOLD}"
    )
    print("=" * 80)

    for i, sample in enumerate(
        samples,
        start=1,
    ):
        result = run_evaluation(
            answer=sample["response"],
            contexts=sample["contexts"],
            reference_answer=None,
        )

        ground_truth = bool(
            sample["adherence_score"]
        )

        prediction = classify_supported(
            result.faithfulness_score
        )

        if prediction and ground_truth:
            tp += 1
        elif not prediction and not ground_truth:
            tn += 1
        elif prediction and not ground_truth:
            fp += 1
        else:
            fn += 1

        print(
            f"{i:02d} | "
            f"ID={sample['id']} | "
            f"GT={ground_truth} | "
            f"Pred={prediction} | "
            f"Faithfulness="
            f"{result.faithfulness_score:.4f}"
        )

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total
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

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"Samples:   {total}")
    print(f"TP:        {tp}")
    print(f"TN:        {tn}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")

    print("-" * 80)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall:    {recall:.4f}"
    )
    print(
        f"F1:        {f1:.4f}"
    )


if __name__ == "__main__":
    main()