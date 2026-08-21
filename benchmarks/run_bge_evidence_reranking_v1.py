
import gc
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from benchmarks.metrics import (
    calculate_binary_metrics,
)


DATASET_NAME = "galileo-ai/ragbench"

VALID_PATH = Path(
    "benchmark_results/"
    "faithfulness_v6_validation.jsonl"
)

OUTPUT_PATH = Path(
    "benchmark_results/"
    "bge_evidence_reranking_v1.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER_MODEL = (
    "BAAI/bge-reranker-v2-m3"
)

NLI_MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

CANDIDATE_TOP_K = 15

SELECTION_K = [
    3,
    5,
]

MAX_LENGTH = 512

RERANK_BATCH_SIZE = 8
NLI_BATCH_SIZE = 16

RRF_CONSTANT = 60


def normalize_key(value):
    return (
        str(value)
        .strip()
        .rstrip(".")
    )


def flatten_evidence(row):
    result = []
    source_order = 0

    for document in (
        row.get("documents_sentences")
        or []
    ):
        for sentence in document:
            if (
                not sentence
                or len(sentence) < 2
            ):
                continue

            key = normalize_key(
                sentence[0]
            )

            text = str(
                sentence[1]
            ).strip()

            if key and text:
                result.append(
                    {
                        "key": key,
                        "text": text,
                        "source_order": (
                            source_order
                        ),
                    }
                )

                source_order += 1

    return result


def best_threshold(
    truths,
    scores,
):
    best_t = None
    best_metrics = None
    best_key = None

    for step in range(
        1,
        1000,
    ):
        threshold = (
            step / 1000.0
        )

        predictions = [
            score >= threshold
            for score in scores
        ]

        metrics = (
            calculate_binary_metrics(
                truths,
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
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_t = threshold
            best_metrics = metrics

    return (
        best_t,
        best_metrics,
    )


def evaluate_scores(
    truths,
    scores,
):
    threshold, metrics = (
        best_threshold(
            truths,
            scores,
        )
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                truths,
                scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                truths,
                scores,
            )
        ),
        "best_threshold": (
            threshold
        ),
        "balanced_accuracy": (
            metrics.balanced_accuracy
        ),
        "precision": (
            metrics.precision
        ),
        "recall": (
            metrics.recall
        ),
        "specificity": (
            metrics.specificity
        ),
        "f1": metrics.f1,
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
    }


def print_result(
    title,
    result,
):
    print()
    print("-" * 80)
    print(title)

    if (
        "any_gold_recall"
        in result
    ):
        print(
            "AnyGold Recall:      ",
            f"{result['any_gold_recall']:.4f}",
        )

        print(
            "AllGold Recall:      ",
            f"{result['all_gold_recall']:.4f}",
        )

    print(
        "ROC-AUC:             ",
        f"{result['roc_auc']:.4f}",
    )

    print(
        "Average Precision:   ",
        f"{result['average_precision']:.4f}",
    )

    print(
        "Best threshold:      ",
        f"{result['best_threshold']:.3f}",
    )

    print(
        "Balanced Accuracy:   ",
        f"{result['balanced_accuracy']:.4f}",
    )

    print(
        "Precision:           ",
        f"{result['precision']:.4f}",
    )

    print(
        "Recall:              ",
        f"{result['recall']:.4f}",
    )

    print(
        "Specificity:         ",
        f"{result['specificity']:.4f}",
    )

    print(
        "F1:                  ",
        f"{result['f1']:.4f}",
    )


# ============================================================
# LOAD FROZEN VALIDATION
# ============================================================

rows = []

with VALID_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        if line.strip():
            rows.append(
                json.loads(line)
            )


if len(rows) != 628:
    raise RuntimeError(
        "Expected 628 frozen "
        "validation samples."
    )


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required."
    )


device = torch.device(
    "cuda"
)


print("=" * 80)
print("BGE EVIDENCE RERANKING V1")
print("=" * 80)

print(
    "Samples:",
    len(rows),
)

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)


# ============================================================
# STAGE 1
# MiniLM candidate retrieval
# ============================================================

print()
print(
    "STAGE 1 — MiniLM Top-15 retrieval"
)


retriever = SentenceTransformer(
    EMBEDDING_MODEL,
    device=str(device),
)


dataset_cache = {}

prepared = []


for index, item in enumerate(
    rows,
    start=1,
):
    subset = item["subset"]

    if subset not in dataset_cache:
        print(
            "Loading:",
            subset,
        )

        dataset_cache[subset] = (
            load_dataset(
                DATASET_NAME,
                subset,
                split="validation",
            )
        )

    source_row = dataset_cache[
        subset
    ][
        int(item["row_index"])
    ]

    evidence = flatten_evidence(
        source_row
    )

    if not evidence:
        raise RuntimeError(
            "Missing source evidence: "
            f"{subset}/"
            f"{item['sample_id']}"
        )

    evidence_texts = [
        entry["text"]
        for entry in evidence
    ]

    claim_vector = (
        retriever.encode(
            [item["claim"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
    )

    evidence_vectors = (
        retriever.encode(
            evidence_texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    similarities = (
        evidence_vectors
        @ claim_vector
    )

    ranking = np.argsort(
        -similarities
    )

    candidates = []

    for embedding_rank, position in enumerate(
        ranking[
            :CANDIDATE_TOP_K
        ],
        start=1,
    ):
        entry = evidence[
            int(position)
        ]

        candidates.append(
            {
                **entry,
                "embedding_rank": (
                    embedding_rank
                ),
                "embedding_score": float(
                    similarities[
                        int(position)
                    ]
                ),
            }
        )

    prepared.append(
        {
            "subset": subset,
            "sample_id": (
                item["sample_id"]
            ),
            "claim": (
                item["claim"]
            ),
            "label": int(
                item["label"]
            ),
            "gold_keys": set(
                item.get(
                    "gold_keys",
                    [],
                )
            ),
            "baseline_evidence": (
                item["evidence"]
            ),
            "candidates": (
                candidates
            ),
        }
    )

    if (
        index % 50 == 0
        or index == len(rows)
    ):
        print(
            f"Retrieved "
            f"{index}/{len(rows)}"
        )


# Release MiniLM before loading
# the much larger cross-encoder.

del retriever

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# STAGE 2
# Dedicated BGE reranker
# ============================================================

print()
print(
    "STAGE 2 — Loading dedicated "
    "BGE reranker..."
)

rerank_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        RERANKER_MODEL
    )
)

rerank_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        RERANKER_MODEL
    )
    .to(device)
)

rerank_model.eval()


rerank_claims = []
rerank_passages = []
rerank_locations = []


for sample_index, item in enumerate(
    prepared
):
    for candidate_index, candidate in enumerate(
        item["candidates"]
    ):
        # Retrieval semantics:
        # query   = claim
        # passage = evidence

        rerank_claims.append(
            item["claim"]
        )

        rerank_passages.append(
            candidate["text"]
        )

        rerank_locations.append(
            (
                sample_index,
                candidate_index,
            )
        )


print(
    "BGE pairs:",
    len(rerank_claims),
)


@torch.no_grad()
def score_bge(
    claims,
    passages,
):
    scores = []

    for start in range(
        0,
        len(claims),
        RERANK_BATCH_SIZE,
    ):
        end = (
            start
            + RERANK_BATCH_SIZE
        )

        encoded = (
            rerank_tokenizer(
                claims[start:end],
                passages[start:end],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = (
                rerank_model(
                    **encoded
                ).logits
            )

        if (
            logits.ndim == 2
            and logits.shape[-1] == 1
        ):
            batch_scores = (
                logits[:, 0]
            )

        elif (
            logits.ndim == 2
            and logits.shape[-1] == 2
        ):
            batch_scores = (
                logits[:, 1]
                - logits[:, 0]
            )

        else:
            raise RuntimeError(
                "Unexpected BGE "
                "reranker logits shape: "
                f"{tuple(logits.shape)}"
            )

        scores.extend(
            batch_scores
            .float()
            .cpu()
            .tolist()
        )

    return scores


bge_scores = score_bge(
    rerank_claims,
    rerank_passages,
)


for (
    sample_index,
    candidate_index,
), score in zip(
    rerank_locations,
    bge_scores,
):
    prepared[
        sample_index
    ][
        "candidates"
    ][
        candidate_index
    ][
        "bge_score"
    ] = float(score)


# Create BGE ranks and
# reciprocal-rank-fusion scores.

for item in prepared:

    bge_order = sorted(
        range(
            len(
                item["candidates"]
            )
        ),
        key=lambda i: (
            item["candidates"][
                i
            ]["bge_score"]
        ),
        reverse=True,
    )

    for bge_rank, candidate_index in enumerate(
        bge_order,
        start=1,
    ):
        candidate = (
            item["candidates"][
                candidate_index
            ]
        )

        candidate[
            "bge_rank"
        ] = bge_rank

        candidate[
            "rrf_score"
        ] = (
            1.0
            / (
                RRF_CONSTANT
                + candidate[
                    "embedding_rank"
                ]
            )
            +
            1.0
            / (
                RRF_CONSTANT
                + bge_rank
            )
        )


print(
    "BGE reranking complete."
)


# Release BGE before NLI judge.

del rerank_model
del rerank_tokenizer

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# STAGE 3
# mDeBERTa judge
# ============================================================

print()
print(
    "STAGE 3 — Loading mDeBERTa judge..."
)


nli_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        NLI_MODEL,
        use_fast=False,
    )
)

nli_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        NLI_MODEL
    )
    .to(device)
)

nli_model.eval()


entailment_index = None

for index, label in (
    nli_model.config.id2label.items()
):
    clean = str(
        label
    ).lower()

    if (
        "entail" in clean
        and "not_entail"
        not in clean
    ):
        entailment_index = int(
            index
        )
        break


if entailment_index is None:
    raise RuntimeError(
        "Could not identify "
        "entailment index."
    )


@torch.no_grad()
def score_nli(
    evidences,
    claims,
):
    scores = []

    for start in range(
        0,
        len(claims),
        NLI_BATCH_SIZE,
    ):
        end = (
            start
            + NLI_BATCH_SIZE
        )

        encoded = (
            nli_tokenizer(
                evidences[start:end],
                claims[start:end],
                padding=True,
                truncation="only_first",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = (
                nli_model(
                    **encoded
                ).logits
            )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
        )

        scores.extend(
            probabilities[
                :,
                entailment_index
            ]
            .float()
            .cpu()
            .tolist()
        )

    return scores


truths = [
    item["label"]
    for item in prepared
]


# ============================================================
# BASELINE
# Exact frozen MiniLM Top-5 evidence
# ============================================================

baseline_scores = score_nli(
    [
        item[
            "baseline_evidence"
        ]
        for item in prepared
    ],
    [
        item["claim"]
        for item in prepared
    ],
)


baseline_result = (
    evaluate_scores(
        truths,
        baseline_scores,
    )
)


print()
print("=" * 80)
print(
    "REFERENCE BASELINE — MINILM TOP-5"
)
print("=" * 80)

print_result(
    "BASELINE",
    baseline_result,
)


if not (
    0.69
    <= baseline_result[
        "roc_auc"
    ]
    <= 0.73
):
    raise RuntimeError(
        "Baseline integrity "
        "check failed."
    )


# ============================================================
# SELECTION STRATEGIES
# ============================================================

results = {
    "baseline_minilm_top5": (
        baseline_result
    )
}


def run_strategy(
    strategy_name,
    ranking_field,
    k,
):
    bundles = []
    claims = []

    gold_eligible = 0
    any_gold_hits = 0
    all_gold_hits = 0


    for item in prepared:

        ranked = sorted(
            item["candidates"],
            key=lambda x: (
                x[ranking_field]
            ),
            reverse=True,
        )

        selected = (
            ranked[:k]
        )

        # Restore original document order
        # after selecting evidence.
        selected = sorted(
            selected,
            key=lambda x: (
                x["source_order"]
            ),
        )

        selected_keys = {
            entry["key"]
            for entry in selected
        }

        if (
            item["label"] == 1
            and item["gold_keys"]
        ):
            gold_eligible += 1

            if (
                item["gold_keys"]
                & selected_keys
            ):
                any_gold_hits += 1

            if (
                item["gold_keys"]
                <= selected_keys
            ):
                all_gold_hits += 1

        bundles.append(
            "\n".join(
                entry["text"]
                for entry in selected
            )
        )

        claims.append(
            item["claim"]
        )


    scores = score_nli(
        bundles,
        claims,
    )

    result = evaluate_scores(
        truths,
        scores,
    )

    result[
        "any_gold_recall"
    ] = (
        any_gold_hits
        / gold_eligible
    )

    result[
        "all_gold_recall"
    ] = (
        all_gold_hits
        / gold_eligible
    )

    results[
        strategy_name
    ] = result

    print_result(
        strategy_name.upper(),
        result,
    )


for k in SELECTION_K:

    run_strategy(
        f"bge_top_{k}",
        "bge_score",
        k,
    )

    run_strategy(
        f"rrf_top_{k}",
        "rrf_score",
        k,
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Baseline AUC:",
    f"{baseline_result['roc_auc']:.4f}",
)


best_name = None
best_result = None


for name, result in results.items():

    if name == "baseline_minilm_top5":
        continue

    if (
        best_result is None
        or result["roc_auc"]
        > best_result["roc_auc"]
    ):
        best_name = name
        best_result = result


print(
    "Best reranking strategy:",
    best_name,
)

print(
    "Best reranking AUC:",
    f"{best_result['roc_auc']:.4f}",
)

print(
    "AUC delta:",
    f"{best_result['roc_auc'] - baseline_result['roc_auc']:+.4f}",
)


OUTPUT_PATH.write_text(
    json.dumps(
        {
            "candidate_top_k": (
                CANDIDATE_TOP_K
            ),
            "rrf_constant": (
                RRF_CONSTANT
            ),
            "embedding_model": (
                EMBEDDING_MODEL
            ),
            "reranker_model": (
                RERANKER_MODEL
            ),
            "nli_model": (
                NLI_MODEL
            ),
            "sample_count": (
                len(prepared)
            ),
            "results": (
                results
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "Saved:",
    OUTPUT_PATH,
)

print("=" * 80)
