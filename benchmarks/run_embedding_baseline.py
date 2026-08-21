import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import load_dataset

from benchmarks.metrics import (
    calculate_binary_metrics,
)
from evaluation.evaluators.faithfulness import (
    FaithfulnessEvaluator,
)
from evaluation.generation.embedding_service import (
    MODEL_NAME,
    get_embedding_model,
)


DEFAULT_MANIFEST = (
    "benchmarks/manifests/"
    "ragbench_validation_v1.json"
)

DEFAULT_THRESHOLD = 0.70


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current production "
            "embedding faithfulness method on "
            "the frozen RAGBench validation manifest."
        )
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
    )

    return parser


def load_manifest(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    manifest = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not manifest.get("samples"):
        raise ValueError(
            "Manifest contains no samples."
        )

    return manifest


def evaluate_samples(
    manifest: dict,
    threshold: float,
) -> tuple[
    list[dict],
    dict,
    dict[str, dict],
]:
    evaluator = FaithfulnessEvaluator()

    print(
        "Loading production embedding model..."
    )
    get_embedding_model()

    dataset_cache = {}

    results: list[dict] = []

    labels: list[bool] = []
    predictions: list[bool] = []

    domain_labels = defaultdict(list)
    domain_predictions = defaultdict(list)

    for index, item in enumerate(
        manifest["samples"],
        start=1,
    ):
        subset = item["subset"]
        split = item["split"]
        row_index = int(
            item["row_index"]
        )

        if subset not in dataset_cache:
            print(
                f"Loading dataset: {subset}"
            )

            dataset_cache[subset] = (
                load_dataset(
                    manifest["dataset"],
                    subset,
                    split=split,
                )
            )

        dataset = dataset_cache[subset]

        row = dataset[row_index]

        current_id = str(
            row["id"]
        )

        expected_id = str(
            item["sample_id"]
        )

        if current_id != expected_id:
            raise RuntimeError(
                "Manifest/data mismatch for "
                f"{subset} row {row_index}: "
                f"expected ID {expected_id}, "
                f"got {current_id}."
            )

        truth = bool(
            item["supported"]
        )

        row_truth = bool(
            row["adherence_score"]
        )

        if truth != row_truth:
            raise RuntimeError(
                "Manifest label mismatch for "
                f"{subset} ID {current_id}."
            )

        contexts = [
            str(document).strip()
            for document
            in (row["documents"] or [])
            if str(document).strip()
        ]

        response = str(
            row["response"]
        ).strip()

        combined_context = "\n\n".join(
            contexts
        )

        started = perf_counter()

        evaluation = evaluator.evaluate(
            answer=response,
            combined_context=combined_context,
        )

        latency_ms = (
            perf_counter() - started
        ) * 1000.0

        prediction = (
            evaluation.score
            >= threshold
        )

        labels.append(truth)
        predictions.append(prediction)

        domain_labels[subset].append(
            truth
        )

        domain_predictions[subset].append(
            prediction
        )

        results.append(
            {
                "subset": subset,
                "split": split,
                "row_index": row_index,
                "sample_id": current_id,
                "truth_supported": truth,
                "predicted_supported": (
                    prediction
                ),
                "score": float(
                    evaluation.score
                ),
                "threshold": threshold,
                "latency_ms": latency_ms,
            }
        )

        if (
            index % 25 == 0
            or index == len(
                manifest["samples"]
            )
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(manifest['samples'])}"
            )

    overall = (
        calculate_binary_metrics(
            labels,
            predictions,
        ).to_dict()
    )

    per_domain = {}

    for subset in manifest["subsets"]:
        metrics = calculate_binary_metrics(
            domain_labels[subset],
            domain_predictions[subset],
        )

        per_domain[subset] = (
            metrics.to_dict()
        )

    return (
        results,
        overall,
        per_domain,
    )


def main() -> None:
    args = build_parser().parse_args()

    if not (
        0.0
        <= args.threshold
        <= 1.0
    ):
        raise ValueError(
            "threshold must be between "
            "0 and 1."
        )

    manifest_path = Path(
        args.manifest
    )

    manifest = load_manifest(
        manifest_path
    )

    (
        sample_results,
        overall_metrics,
        per_domain_metrics,
    ) = evaluate_samples(
        manifest=manifest,
        threshold=args.threshold,
    )

    latencies = [
        item["latency_ms"]
        for item in sample_results
    ]

    payload = {
        "benchmark_version": (
            manifest[
                "benchmark_version"
            ]
        ),
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "sample_count": len(
            sample_results
        ),
        "supported_count": (
            manifest[
                "supported_count"
            ]
        ),
        "unsupported_count": (
            manifest[
                "unsupported_count"
            ]
        ),
        "evaluator": (
            "production_embedding_"
            "faithfulness"
        ),
        "model": MODEL_NAME,
        "threshold": args.threshold,
        "overall_metrics": (
            overall_metrics
        ),
        "per_domain_metrics": (
            per_domain_metrics
        ),
        "latency_ms": {
            "mean": float(
                np.mean(latencies)
            ),
            "p50": float(
                np.percentile(
                    latencies,
                    50,
                )
            ),
            "p95": float(
                np.percentile(
                    latencies,
                    95,
                )
            ),
            "max": float(
                np.max(latencies)
            ),
        },
        "samples": sample_results,
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "embedding_ragbench_"
        "validation_v1.json"
    )

    output_path.write_text(
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
        "PRODUCTION EMBEDDING "
        "FAITHFULNESS BASELINE"
    )
    print("=" * 76)

    print(
        "Benchmark:",
        manifest[
            "benchmark_version"
        ],
    )
    print(
        "Samples:  ",
        len(sample_results),
    )
    print(
        "Model:    ",
        MODEL_NAME,
    )
    print(
        "Threshold:",
        f"{args.threshold:.4f}",
    )

    print("-" * 76)

    for name in [
        "tp",
        "tn",
        "fp",
        "fn",
    ]:
        print(
            f"{name.upper():<20}",
            overall_metrics[name],
        )

    print("-" * 76)

    metric_names = [
        (
            "Accuracy",
            "accuracy",
        ),
        (
            "Balanced Accuracy",
            "balanced_accuracy",
        ),
        (
            "Precision",
            "precision",
        ),
        (
            "Recall",
            "recall",
        ),
        (
            "Specificity",
            "specificity",
        ),
        (
            "F1",
            "f1",
        ),
    ]

    for label, key in metric_names:
        print(
            f"{label:<20}"
            f"{overall_metrics[key]:.4f}"
        )

    print()
    print("=" * 76)
    print("PER-DOMAIN BALANCED ACCURACY")
    print("=" * 76)

    for subset in manifest["subsets"]:
        metrics = (
            per_domain_metrics[
                subset
            ]
        )

        print(
            f"{subset:<14}"
            f" BA="
            f"{metrics['balanced_accuracy']:.4f}"
            f" Recall="
            f"{metrics['recall']:.4f}"
            f" Spec="
            f"{metrics['specificity']:.4f}"
            f" F1="
            f"{metrics['f1']:.4f}"
        )

    print("-" * 76)

    print(
        "Mean latency:",
        f"{payload['latency_ms']['mean']:.2f} ms",
    )

    print(
        "P95 latency: ",
        f"{payload['latency_ms']['p95']:.2f} ms",
    )

    print(
        "Saved:       ",
        output_path,
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
