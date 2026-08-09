from benchmarks.ragbench_loader import load_ragbench_samples
from evaluation.evaluators.faithfulness import FaithfulnessEvaluator
from evaluation.evaluators.faithfulness_nli import NLIFaithfulnessEvaluator


SAMPLE_LIMIT = 20
V1_THRESHOLD = 0.75


def calculate_metrics(predictions, labels):
    tp = tn = fp = fn = 0

    for prediction, label in zip(predictions, labels):
        if prediction and label:
            tp += 1
        elif not prediction and not label:
            tn += 1
        elif prediction and not label:
            fp += 1
        else:
            fn += 1

    total = len(labels)

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


def print_metrics(name, metrics):
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"TP: {metrics['tp']}")
    print(f"TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")

    print("-" * 80)

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Precision:         "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall:            "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"Specificity:       "
        f"{metrics['specificity']:.4f}"
    )
    print(
        f"F1:                "
        f"{metrics['f1']:.4f}"
    )


def main():
    print(
        f"Loading {SAMPLE_LIMIT} "
        "RAGBench validation samples..."
    )

    samples = load_ragbench_samples(
        split="validation",
        limit=SAMPLE_LIMIT,
    )

    embedding_evaluator = (
        FaithfulnessEvaluator()
    )

    nli_evaluator = (
        NLIFaithfulnessEvaluator()
    )

    labels = []

    embedding_predictions = []
    nli_predictions = []

    for i, sample in enumerate(
        samples,
        start=1,
    ):
        label = bool(
            sample["adherence_score"]
        )

        labels.append(label)

        embedding_result = (
            embedding_evaluator.evaluate(
                answer=sample["response"],
                contexts=sample["contexts"],
            )
        )

        embedding_prediction = (
            embedding_result.score
            >= V1_THRESHOLD
        )

        embedding_predictions.append(
            embedding_prediction
        )

        nli_result = (
            nli_evaluator.evaluate(
                answer=sample["response"],
                contexts=sample["contexts"],
            )
        )

        # Strict faithfulness:
        # every answer claim must be entailed
        # by retrieved evidence.
        nli_prediction = (
            bool(nli_result.claims)
            and all(
                claim.label == "entailment"
                for claim
                in nli_result.claims
            )
        )

        nli_predictions.append(
            nli_prediction
        )

        print("=" * 80)

        print(
            f"{i:02d} | "
            f"ID={sample['id']} | "
            f"GT={label}"
        )

        print(
            "Embedding:"
            f" score={embedding_result.score:.4f}"
            f" pred={embedding_prediction}"
        )

        print(
            "NLI:"
            f" score={nli_result.score:.4f}"
            f" pred={nli_prediction}"
        )

        for claim in nli_result.claims:
            print(
                "  Claim:",
                claim.claim,
            )

            print(
                "  NLI:",
                claim.label,
                "| entailment=",
                claim.entailment,
                "| contradiction=",
                claim.contradiction,
            )

    embedding_metrics = (
        calculate_metrics(
            embedding_predictions,
            labels,
        )
    )

    nli_metrics = (
        calculate_metrics(
            nli_predictions,
            labels,
        )
    )

    print_metrics(
        "V1 - EMBEDDING FAITHFULNESS",
        embedding_metrics,
    )

    print_metrics(
        "V2 - NLI FAITHFULNESS",
        nli_metrics,
    )


if __name__ == "__main__":
    main()