import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from benchmarks.metrics import (
    calculate_binary_metrics,
)
from benchmarks.nli_model import (
    MODEL_NAME,
    NLIEntailmentModel,
)
from benchmarks.nli_scoring import (
    classify_score,
    strict_answer_score,
)


DEFAULT_MANIFEST = (
    "benchmarks/manifests/"
    "ragbench_validation_v1.json"
)

DEFAULT_THRESHOLD = 0.50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pretrained multilingual "
            "DeBERTa NLI on the frozen "
            "RAGBench validation benchmark."
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
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
    )

    return parser


def clean_text_list(
    values,
) -> list[str]:
    if not values:
        return []

    return [
        str(value).strip()
        for value in values
        if value
        and str(value).strip()
    ]


def calculate_claim_support(
    model: NLIEntailmentModel,
    claims: list[str],
    documents: list[str],
    batch_size: int,
) -> list[float]:
    if not claims or not documents:
        return []

    pairs = [
        (
            document,
            claim,
        )
        for claim in claims
        for document in documents
    ]

    pair_scores = (
        model.entailment_scores(
            pairs,
            batch_size=batch_size,
        )
    )

    document_count = len(
        documents
    )

    claim_scores = []

    for claim_index in range(
        len(claims)
    ):
        start = (
            claim_index
            * document_count
        )

        end = (
            start
            + document_count
        )

        evidence_scores = (
            pair_scores[
                start:end
            ]
        )

        claim_scores.append(
            max(evidence_scores)
        )

    return claim_scores


def main() -> None:
    args = build_parser().parse_args()

    if not (
        0.0
        <= args.threshold
        <= 1.0
    ):
        raise ValueError(
            "threshold must be "
            "between 0 and 1."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be "
            "greater than zero."
        )

    manifest_path = Path(
        args.manifest
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manifest_samples = list(
        manifest["samples"]
    )

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(
                "max-samples must be "
                "greater than zero."
            )

        manifest_samples = (
            manifest_samples[
                :args.max_samples
            ]
        )

    model = NLIEntailmentModel()

    print(
        "Device:",
        model.info.device,
    )

    dataset_cache = {}

    labels = []
    predictions = []
    scores = []
    latencies = []
    sample_results = []

    domain_labels = defaultdict(
        list
    )

    domain_predictions = defaultdict(
        list
    )

    total = len(
        manifest_samples
    )

    for index, item in enumerate(
        manifest_samples,
        start=1,
    ):
        subset = item["subset"]
        split = item["split"]
        row_index = int(
            item["row_index"]
        )

        if subset not in dataset_cache:
            print(
                f"Loading dataset: "
                f"{subset}"
            )

            dataset_cache[subset] = (
                load_dataset(
                    manifest["dataset"],
                    subset,
                    split=split,
                )
            )

        row = (
            dataset_cache[
                subset
            ][row_index]
        )

        sample_id = str(
            row["id"]
        )

        if sample_id != str(
            item["sample_id"]
        ):
            raise RuntimeError(
                "Manifest/data ID "
                "mismatch."
            )

        truth = bool(
            item["supported"]
        )

        if truth != bool(
            row["adherence_score"]
        ):
            raise RuntimeError(
                "Manifest/data label "
                "mismatch."
            )

        documents = clean_text_list(
            row.get(
                "documents"
            )
        )

        claims = clean_text_list(
            row.get(
                "response_sentences"
            )
        )

        if not claims:
            response = str(
                row["response"]
            ).strip()

            claims = (
                [response]
                if response
                else []
            )

        started = perf_counter()

        claim_scores = (
            calculate_claim_support(
                model=model,
                claims=claims,
                documents=documents,
                batch_size=(
                    args.batch_size
                ),
            )
        )

        answer_score = (
            strict_answer_score(
                claim_scores
            )
        )

        latency_ms = (
            perf_counter()
            - started
        ) * 1000.0

        prediction = (
            classify_score(
                answer_score,
                args.threshold,
            )
        )

        labels.append(
            truth
        )

        predictions.append(
            prediction
        )

        scores.append(
            answer_score
        )

        latencies.append(
            latency_ms
        )

        domain_labels[
            subset
        ].append(
            truth
        )

        domain_predictions[
            subset
        ].append(
            prediction
        )

        sample_results.append(
            {
                "subset": subset,
                "split": split,
                "row_index": (
                    row_index
                ),
                "sample_id": (
                    sample_id
                ),
                "truth_supported": (
                    truth
                ),
                "predicted_supported": (
                    prediction
                ),
                "score": (
                    answer_score
                ),
                "threshold": (
                    args.threshold
                ),
                "claim_count": len(
                    claims
                ),
                "document_count": len(
                    documents
                ),
                "claim_scores": [
                    float(value)
                    for value
                    in claim_scores
                ],
                "latency_ms": (
                    latency_ms
                ),
            }
        )

        if (
            index % 10 == 0
            or index == total
        ):
            print(
                f"Processed "
                f"{index}/{total}"
            )

    metrics = (
        calculate_binary_metrics(
            labels,
            predictions,
        )
    )

    roc_auc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    average_precision = float(
        average_precision_score(
            labels,
            scores,
        )
    )

    per_domain = {}

    for subset in (
        domain_labels.keys()
    ):
        domain_metrics = (
            calculate_binary_metrics(
                domain_labels[
                    subset
                ],
                domain_predictions[
                    subset
                ],
            )
        )

        per_domain[subset] = (
            domain_metrics.to_dict()
        )

    payload = {
        "benchmark_version": (
            manifest[
                "benchmark_version"
            ]
        ),
        "dataset": (
            manifest["dataset"]
        ),
        "sample_count": total,
        "evaluator": (
            "pretrained_mdeberta_nli"
        ),
        "model": MODEL_NAME,
        "device": (
            model.info.device
        ),
        "threshold": (
            args.threshold
        ),
        "aggregation": (
            "min_over_claims_of_"
            "max_entailment_over_"
            "retrieved_documents"
        ),
        "metrics": (
            metrics.to_dict()
        ),
        "roc_auc": roc_auc,
        "average_precision": (
            average_precision
        ),
        "latency_ms": {
            "mean": float(
                np.mean(
                    latencies
                )
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
        },
        "per_domain_metrics": (
            per_domain
        ),
        "samples": (
            sample_results
        ),
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "full"
        if args.max_samples is None
        else str(
            args.max_samples
        )
    )

    output_path = (
        output_dir
        / (
            "mdeberta_ragbench_"
            f"validation_v1_{suffix}.json"
        )
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
        "PRETRAINED mDeBERTa "
        "NLI BASELINE"
    )
    print("=" * 76)

    print(
        "Samples:             ",
        total,
    )

    print(
        "Model:               ",
        MODEL_NAME,
    )

    print(
        "Device:              ",
        model.info.device,
    )

    print(
        "Threshold:           ",
        f"{args.threshold:.2f}",
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

    print("-" * 76)

    print(
        "ROC-AUC:             ",
        f"{roc_auc:.4f}",
    )

    print(
        "Average Precision:   ",
        f"{average_precision:.4f}",
    )

    print()
    print("=" * 76)
    print(
        "PER-DOMAIN RESULTS"
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
        "Mean latency:        ",
        f"{payload['latency_ms']['mean']:.2f} ms",
    )

    print(
        "P95 latency:         ",
        f"{payload['latency_ms']['p95']:.2f} ms",
    )

    print(
        "Saved:               ",
        output_path,
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
