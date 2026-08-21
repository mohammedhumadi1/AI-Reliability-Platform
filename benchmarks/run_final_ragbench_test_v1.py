
import gc
import hashlib
import json
import random
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


# ============================================================
# FROZEN FINAL TEST CONFIGURATION
# ============================================================

DATASET_NAME = "galileo-ai/ragbench"
TEST_SPLIT = "test"

DOMAINS = [
    "covidqa",
    "cuad",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]

SAMPLES_PER_CLASS_PER_DOMAIN = 25
SEED = 20260821

EXPECTED_CANDIDATE_VERSION = (
    "faithfulness-bge-v1"
)

EXPECTED_CANDIDATE_THRESHOLD = 0.254

CANDIDATE_FREEZE_PATH = Path(
    "/content/drive/MyDrive/"
    "AI-Reliability-Platform/"
    "faithfulness_final_candidate_v1/"
    "faithfulness_candidate_v1.json"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/"
    "AI-Reliability-Platform/"
    "faithfulness_final_holdout_v1"
)

RESULT_PATH = (
    OUTPUT_DIR
    / "ragbench_final_test_result.json"
)

TEST_MANIFEST_PATH = (
    OUTPUT_DIR
    / "ragbench_test_manifest.json"
)

TEST_CLAIMS_PATH = (
    OUTPUT_DIR
    / "ragbench_test_claims.jsonl"
)

TEST_SCORES_PATH = (
    OUTPUT_DIR
    / "ragbench_test_scores.jsonl"
)

BOOTSTRAP_REPS = 2000
MAX_LENGTH = 512

EMBED_BATCH_SIZE = 64
BGE_BATCH_SIZE = 8
NLI_BATCH_SIZE = 16


# ============================================================
# SAFETY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if RESULT_PATH.exists():
    raise RuntimeError(
        "FINAL TEST RESULT ALREADY EXISTS. "
        "Refusing to rerun the holdout."
    )


if not CANDIDATE_FREEZE_PATH.exists():
    raise FileNotFoundError(
        CANDIDATE_FREEZE_PATH
    )


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_key(value):
    return (
        str(value)
        .strip()
        .rstrip(".")
    )


def stable_subset_seed(subset):
    raw = (
        f"{SEED}:{subset}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        raw
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def stable_negative_key(record):
    raw = (
        f"{SEED}:"
        f"{record['subset']}:"
        f"{record['sample_id']}:"
        f"{record['claim_key']}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


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


def get_claims(row):
    result = []

    for sentence in (
        row.get("response_sentences")
        or []
    ):
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
                (
                    key,
                    text,
                )
            )

    return result


def get_support_map(row):
    result = {}

    for info in (
        row.get(
            "sentence_support_information"
        )
        or []
    ):
        key = normalize_key(
            info.get(
                "response_sentence_key",
                "",
            )
        )

        if key:
            result[key] = info

    return result


def fixed_threshold_metrics(
    truths,
    scores,
    threshold,
):
    predictions = [
        score >= threshold
        for score in scores
    ]

    return (
        calculate_binary_metrics(
            [
                bool(value)
                for value in truths
            ],
            predictions,
        )
    )


def metrics_dict(metrics):
    return {
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "accuracy": (
            metrics.accuracy
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
    }


def percentile_interval(values):
    return [
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    ]


# ============================================================
# VERIFY FROZEN CANDIDATE
# ============================================================

freeze = json.loads(
    CANDIDATE_FREEZE_PATH.read_text(
        encoding="utf-8"
    )
)


if (
    freeze["candidate_version"]
    != EXPECTED_CANDIDATE_VERSION
):
    raise RuntimeError(
        "Candidate version mismatch."
    )


if (
    freeze["status"]
    != "FROZEN_BEFORE_FINAL_TEST"
):
    raise RuntimeError(
        "Candidate is not in frozen status."
    )


architecture = freeze[
    "architecture"
]


if architecture[
    "candidate_top_k"
] != 15:
    raise RuntimeError(
        "Frozen candidate Top-K mismatch."
    )


if architecture[
    "reranker_selection_k"
] != 5:
    raise RuntimeError(
        "Frozen BGE selection K mismatch."
    )


if architecture[
    "reranking_strategy"
] != "BGE score":
    raise RuntimeError(
        "Frozen reranking strategy mismatch."
    )


if architecture[
    "judge_fine_tuned"
]:
    raise RuntimeError(
        "Frozen candidate unexpectedly "
        "uses a fine-tuned judge."
    )


candidate_threshold = float(
    freeze[
        "decision_threshold"
    ][
        "value"
    ]
)


if (
    abs(
        candidate_threshold
        - EXPECTED_CANDIDATE_THRESHOLD
    )
    > 1e-9
):
    raise RuntimeError(
        "Frozen threshold mismatch."
    )


baseline_threshold = float(
    freeze[
        "development_validation"
    ][
        "baseline"
    ][
        "best_threshold"
    ]
)


EMBEDDING_MODEL = (
    architecture[
        "candidate_retriever"
    ]
)

BGE_MODEL = (
    architecture[
        "reranker"
    ]
)

NLI_MODEL = (
    architecture[
        "judge"
    ]
)


print("=" * 80)
print("FINAL RAGBENCH TEST HOLDOUT")
print("=" * 80)

print(
    "Candidate:",
    freeze["candidate_version"],
)

print(
    "Architecture:",
    "MiniLM Top-15 -> "
    "BGE Top-5 -> "
    "mDeBERTa",
)

print(
    "Frozen candidate threshold:",
    candidate_threshold,
)

print(
    "Frozen baseline threshold:",
    baseline_threshold,
)

print(
    "Candidate freeze SHA256:",
    sha256_file(
        CANDIDATE_FREEZE_PATH
    ),
)

print()
print(
    "No threshold tuning will be "
    "performed on TEST."
)


# ============================================================
# BUILD TEST SAMPLE MANIFEST
#
# This mirrors the frozen validation design:
# 25 supported + 25 unsupported samples/domain.
# ============================================================

dataset_cache = {}

manifest_samples = []
manifest_summary = {}


print()
print("=" * 80)
print("BUILDING FINAL TEST MANIFEST")
print("=" * 80)


for subset in DOMAINS:
    print(
        "Loading TEST:",
        subset,
    )

    dataset = load_dataset(
        DATASET_NAME,
        subset,
        split=TEST_SPLIT,
    )

    dataset_cache[
        subset
    ] = dataset

    labels = [
        bool(value)
        for value
        in dataset[
            "adherence_score"
        ]
    ]

    supported = [
        index
        for index, label
        in enumerate(labels)
        if label
    ]

    unsupported = [
        index
        for index, label
        in enumerate(labels)
        if not label
    ]


    if (
        len(supported)
        < SAMPLES_PER_CLASS_PER_DOMAIN
    ):
        raise RuntimeError(
            f"{subset}: fewer than "
            f"{SAMPLES_PER_CLASS_PER_DOMAIN} "
            "supported TEST samples."
        )

    if (
        len(unsupported)
        < SAMPLES_PER_CLASS_PER_DOMAIN
    ):
        raise RuntimeError(
            f"{subset}: fewer than "
            f"{SAMPLES_PER_CLASS_PER_DOMAIN} "
            "unsupported TEST samples."
        )


    rng = random.Random(
        stable_subset_seed(
            subset
        )
    )

    selected_supported = rng.sample(
        supported,
        SAMPLES_PER_CLASS_PER_DOMAIN,
    )

    selected_unsupported = rng.sample(
        unsupported,
        SAMPLES_PER_CLASS_PER_DOMAIN,
    )

    selected_indices = sorted(
        selected_supported
        + selected_unsupported
    )


    subset_supported = 0
    subset_unsupported = 0


    for row_index in selected_indices:
        row = dataset[
            row_index
        ]

        label = bool(
            row["adherence_score"]
        )

        if label:
            subset_supported += 1
        else:
            subset_unsupported += 1

        manifest_samples.append(
            {
                "subset": subset,
                "split": TEST_SPLIT,
                "row_index": (
                    row_index
                ),
                "sample_id": str(
                    row["id"]
                ),
                "supported": label,
            }
        )


    manifest_summary[
        subset
    ] = {
        "total": (
            len(selected_indices)
        ),
        "supported": (
            subset_supported
        ),
        "unsupported": (
            subset_unsupported
        ),
    }


if len(
    manifest_samples
) != 500:
    raise RuntimeError(
        "Expected exactly 500 "
        "sample-level TEST rows."
    )


test_manifest = {
    "dataset": DATASET_NAME,
    "split": TEST_SPLIT,
    "seed": SEED,
    "selection_strategy": (
        "10 frozen domains; "
        "25 supported and "
        "25 unsupported samples "
        "per domain."
    ),
    "sample_count": (
        len(manifest_samples)
    ),
    "domains": DOMAINS,
    "summary": (
        manifest_summary
    ),
    "samples": (
        manifest_samples
    ),
}


TEST_MANIFEST_PATH.write_text(
    json.dumps(
        test_manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print(
    "Sample-level TEST rows:",
    len(manifest_samples),
)


# ============================================================
# BUILD CLAIM-LEVEL TEST BENCHMARK
#
# Same rules as frozen V6 validation:
# - explicit real gold for positives
# - exclude pseudo-key positives
# - deterministic unsupported balancing
# ============================================================

positive_claims = []
negative_claims = []

skipped_missing_info = 0
skipped_special_positive = 0


for item in manifest_samples:
    subset = item[
        "subset"
    ]

    row = dataset_cache[
        subset
    ][
        int(
            item["row_index"]
        )
    ]


    if (
        str(row["id"])
        != item["sample_id"]
    ):
        raise RuntimeError(
            "TEST manifest ID mismatch."
        )


    evidence = flatten_evidence(
        row
    )

    evidence_keys = {
        entry["key"]
        for entry in evidence
    }

    support_map = get_support_map(
        row
    )


    for (
        claim_key,
        claim_text,
    ) in get_claims(row):

        info = support_map.get(
            claim_key
        )

        if info is None:
            skipped_missing_info += 1
            continue


        fully_supported = bool(
            info.get(
                "fully_supported",
                False,
            )
        )


        support_keys = [
            normalize_key(key)
            for key in (
                info.get(
                    "supporting_sentence_keys"
                )
                or []
            )
        ]


        real_support_keys = [
            key
            for key in support_keys
            if key in evidence_keys
        ]


        record = {
            "subset": subset,
            "sample_id": (
                item["sample_id"]
            ),
            "row_index": int(
                item["row_index"]
            ),
            "claim_key": (
                claim_key
            ),
            "claim": (
                claim_text
            ),
            "gold_keys": (
                real_support_keys
            ),
        }


        if fully_supported:
            if (
                not support_keys
                or len(
                    real_support_keys
                )
                != len(
                    support_keys
                )
            ):
                skipped_special_positive += 1
                continue

            record[
                "label"
            ] = 1

            positive_claims.append(
                record
            )

        else:
            record[
                "label"
            ] = 0

            negative_claims.append(
                record
            )


negative_claims.sort(
    key=stable_negative_key
)


if (
    len(negative_claims)
    < len(positive_claims)
):
    raise RuntimeError(
        "Not enough unsupported "
        "claims to mirror the "
        "frozen validation design."
    )


negative_claims = (
    negative_claims[
        :len(positive_claims)
    ]
)


test_claims = (
    positive_claims
    + negative_claims
)


if not positive_claims:
    raise RuntimeError(
        "No supported TEST claims."
    )


with TEST_CLAIMS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    for row in test_claims:
        file.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "Claim-level TEST rows:",
    len(test_claims),
)

print(
    "Supported claims:",
    len(positive_claims),
)

print(
    "Unsupported claims:",
    len(negative_claims),
)

print(
    "Skipped missing info:",
    skipped_missing_info,
)

print(
    "Skipped special positives:",
    skipped_special_positive,
)


# ============================================================
# STAGE 1 — MiniLM Top-15 Candidate Retrieval
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU required."
    )


device = torch.device(
    "cuda"
)


print()
print("=" * 80)
print("STAGE 1 — MINILM RETRIEVAL")
print("=" * 80)


retriever = SentenceTransformer(
    EMBEDDING_MODEL,
    device=str(device),
)


prepared = []


for index, item in enumerate(
    test_claims,
    start=1,
):
    row = dataset_cache[
        item["subset"]
    ][
        item["row_index"]
    ]

    evidence = flatten_evidence(
        row
    )

    if not evidence:
        raise RuntimeError(
            "Missing TEST evidence."
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
            batch_size=(
                EMBED_BATCH_SIZE
            ),
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
        ranking[:15],
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


    baseline_selected = (
        candidates[:5]
    )

    # Baseline reproduces the old
    # embedding-ranked Top-5 order.
    baseline_bundle = "\n".join(
        entry["text"]
        for entry
        in baseline_selected
    )


    prepared.append(
        {
            **item,
            "candidates": (
                candidates
            ),
            "baseline_bundle": (
                baseline_bundle
            ),
        }
    )


    if (
        index % 50 == 0
        or index == len(
            test_claims
        )
    ):
        print(
            f"Retrieved "
            f"{index}/"
            f"{len(test_claims)}"
        )


del retriever

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# STAGE 2 — Frozen BGE Reranker
# ============================================================

print()
print("=" * 80)
print("STAGE 2 — BGE TOP-5 RERANKING")
print("=" * 80)


bge_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        BGE_MODEL
    )
)

bge_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BGE_MODEL
    )
    .to(device)
)

bge_model.eval()


claims_for_bge = []
passages_for_bge = []
locations = []


for sample_index, item in enumerate(
    prepared
):
    for candidate_index, candidate in enumerate(
        item["candidates"]
    ):
        claims_for_bge.append(
            item["claim"]
        )

        passages_for_bge.append(
            candidate["text"]
        )

        locations.append(
            (
                sample_index,
                candidate_index,
            )
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
        BGE_BATCH_SIZE,
    ):
        end = (
            start
            + BGE_BATCH_SIZE
        )

        encoded = bge_tokenizer(
            claims[start:end],
            passages[start:end],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
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
                bge_model(
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
                "Unexpected BGE logits "
                f"shape: "
                f"{tuple(logits.shape)}"
            )


        scores.extend(
            batch_scores
            .float()
            .cpu()
            .tolist()
        )

    return scores


print(
    "BGE candidate pairs:",
    len(claims_for_bge),
)


bge_scores = score_bge(
    claims_for_bge,
    passages_for_bge,
)


for (
    sample_index,
    candidate_index,
), score in zip(
    locations,
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
    ] = float(
        score
    )


candidate_any_gold = 0
candidate_all_gold = 0

baseline_any_gold = 0
baseline_all_gold = 0

gold_eligible = 0


for item in prepared:
    bge_ranked = sorted(
        item["candidates"],
        key=lambda x: (
            x["bge_score"]
        ),
        reverse=True,
    )

    selected = (
        bge_ranked[:5]
    )

    # Frozen candidate restores
    # source/document order after
    # BGE selection.
    selected_source_order = sorted(
        selected,
        key=lambda x: (
            x["source_order"]
        ),
    )


    item[
        "candidate_bundle"
    ] = "\n".join(
        entry["text"]
        for entry
        in selected_source_order
    )


    if (
        item["label"] == 1
        and item["gold_keys"]
    ):
        gold_eligible += 1

        gold_keys = set(
            item["gold_keys"]
        )

        candidate_keys = {
            entry["key"]
            for entry in selected
        }

        baseline_keys = {
            entry["key"]
            for entry
            in item["candidates"][:5]
        }


        if (
            gold_keys
            & candidate_keys
        ):
            candidate_any_gold += 1

        if (
            gold_keys
            <= candidate_keys
        ):
            candidate_all_gold += 1


        if (
            gold_keys
            & baseline_keys
        ):
            baseline_any_gold += 1

        if (
            gold_keys
            <= baseline_keys
        ):
            baseline_all_gold += 1


del bge_model
del bge_tokenizer

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# STAGE 3 — Frozen mDeBERTa Judge
# ============================================================

print()
print("=" * 80)
print("STAGE 3 — mDeBERTa JUDGE")
print("=" * 80)


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
        "Entailment label not found."
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

        encoded = nli_tokenizer(
            evidences[start:end],
            claims[start:end],
            padding=True,
            truncation="only_first",
            max_length=MAX_LENGTH,
            return_tensors="pt",
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
    int(item["label"])
    for item in prepared
]

claims = [
    item["claim"]
    for item in prepared
]


baseline_scores = score_nli(
    [
        item["baseline_bundle"]
        for item in prepared
    ],
    claims,
)


candidate_scores = score_nli(
    [
        item["candidate_bundle"]
        for item in prepared
    ],
    claims,
)


# ============================================================
# FINAL METRICS
#
# NO TEST THRESHOLD SEARCH.
# ============================================================

baseline_auc = float(
    roc_auc_score(
        truths,
        baseline_scores,
    )
)

candidate_auc = float(
    roc_auc_score(
        truths,
        candidate_scores,
    )
)


baseline_ap = float(
    average_precision_score(
        truths,
        baseline_scores,
    )
)

candidate_ap = float(
    average_precision_score(
        truths,
        candidate_scores,
    )
)


baseline_metrics = (
    fixed_threshold_metrics(
        truths,
        baseline_scores,
        baseline_threshold,
    )
)


candidate_metrics = (
    fixed_threshold_metrics(
        truths,
        candidate_scores,
        candidate_threshold,
    )
)


auc_delta = (
    candidate_auc
    - baseline_auc
)


# ============================================================
# PAIRED BOOTSTRAP CI
#
# Diagnostic uncertainty only.
# Does NOT tune or select anything.
# ============================================================

truth_array = np.asarray(
    truths,
    dtype=np.int64,
)

baseline_array = np.asarray(
    baseline_scores,
    dtype=np.float64,
)

candidate_array = np.asarray(
    candidate_scores,
    dtype=np.float64,
)


rng = np.random.default_rng(
    SEED
)


bootstrap_baseline = []
bootstrap_candidate = []
bootstrap_delta = []


n = len(
    truth_array
)


for _ in range(
    BOOTSTRAP_REPS
):
    indices = rng.integers(
        0,
        n,
        size=n,
    )

    y = truth_array[
        indices
    ]

    # Rare defensive case.
    if (
        np.unique(y).size < 2
    ):
        continue


    base_auc = roc_auc_score(
        y,
        baseline_array[
            indices
        ],
    )

    cand_auc = roc_auc_score(
        y,
        candidate_array[
            indices
        ],
    )


    bootstrap_baseline.append(
        base_auc
    )

    bootstrap_candidate.append(
        cand_auc
    )

    bootstrap_delta.append(
        cand_auc
        - base_auc
    )


baseline_auc_ci = (
    percentile_interval(
        bootstrap_baseline
    )
)

candidate_auc_ci = (
    percentile_interval(
        bootstrap_candidate
    )
)

delta_auc_ci = (
    percentile_interval(
        bootstrap_delta
    )
)


# ============================================================
# WRITE PER-CLAIM SCORES
# ============================================================

with TEST_SCORES_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    for (
        item,
        baseline_score,
        candidate_score,
    ) in zip(
        prepared,
        baseline_scores,
        candidate_scores,
    ):
        file.write(
            json.dumps(
                {
                    "subset": (
                        item["subset"]
                    ),
                    "sample_id": (
                        item["sample_id"]
                    ),
                    "claim_key": (
                        item["claim_key"]
                    ),
                    "label": (
                        item["label"]
                    ),
                    "baseline_score": float(
                        baseline_score
                    ),
                    "candidate_score": float(
                        candidate_score
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# INTERPRETATION
# ============================================================

if (
    auc_delta > 0
    and delta_auc_ci[0] > 0
):
    generalization_status = (
        "STRONG_POSITIVE_GENERALIZATION"
    )

elif auc_delta > 0:
    generalization_status = (
        "POSITIVE_BUT_CI_INCLUDES_ZERO"
    )

else:
    generalization_status = (
        "NO_POSITIVE_GENERALIZATION"
    )


result = {
    "evaluation_type": (
        "FINAL_ONE_SHOT_RAGBENCH_TEST"
    ),

    "candidate_version": (
        freeze[
            "candidate_version"
        ]
    ),

    "candidate_freeze_sha256": (
        sha256_file(
            CANDIDATE_FREEZE_PATH
        )
    ),

    "test_manifest_sha256": (
        sha256_file(
            TEST_MANIFEST_PATH
        )
    ),

    "test_claims_sha256": (
        sha256_file(
            TEST_CLAIMS_PATH
        )
    ),

    "test_split": (
        TEST_SPLIT
    ),

    "sample_level_count": (
        len(manifest_samples)
    ),

    "claim_level_count": (
        len(test_claims)
    ),

    "supported_claims": (
        len(positive_claims)
    ),

    "unsupported_claims": (
        len(negative_claims)
    ),

    "threshold_policy": {
        "candidate_threshold": (
            candidate_threshold
        ),
        "baseline_threshold": (
            baseline_threshold
        ),
        "threshold_tuned_on_test": (
            False
        ),
    },

    "retrieval_diagnostics": {
        "gold_eligible": (
            gold_eligible
        ),

        "baseline_any_gold_recall": (
            baseline_any_gold
            / gold_eligible
        ),

        "baseline_all_gold_recall": (
            baseline_all_gold
            / gold_eligible
        ),

        "candidate_any_gold_recall": (
            candidate_any_gold
            / gold_eligible
        ),

        "candidate_all_gold_recall": (
            candidate_all_gold
            / gold_eligible
        ),
    },

    "baseline": {
        "roc_auc": (
            baseline_auc
        ),
        "roc_auc_95_ci": (
            baseline_auc_ci
        ),
        "average_precision": (
            baseline_ap
        ),
        "fixed_threshold": (
            baseline_threshold
        ),
        "metrics": (
            metrics_dict(
                baseline_metrics
            )
        ),
    },

    "candidate": {
        "roc_auc": (
            candidate_auc
        ),
        "roc_auc_95_ci": (
            candidate_auc_ci
        ),
        "average_precision": (
            candidate_ap
        ),
        "fixed_threshold": (
            candidate_threshold
        ),
        "metrics": (
            metrics_dict(
                candidate_metrics
            )
        ),
    },

    "comparison": {
        "auc_delta": (
            auc_delta
        ),
        "auc_delta_95_ci": (
            delta_auc_ci
        ),
        "status": (
            generalization_status
        ),
    },

    "rules": {
        "architecture_changed_on_test": (
            False
        ),
        "threshold_retuned_on_test": (
            False
        ),
        "rrf_evaluated_for_selection": (
            False
        ),
        "v6_adapter_used": (
            False
        ),
    },
}


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL HOLDOUT RESULT")
print("=" * 80)

print(
    "Sample-level rows:",
    len(manifest_samples),
)

print(
    "Claim-level rows:",
    len(test_claims),
)

print(
    "Supported:",
    len(positive_claims),
)

print(
    "Unsupported:",
    len(negative_claims),
)


print()
print("BASELINE — MiniLM Top-5")
print("-" * 80)

print(
    "ROC-AUC:",
    f"{baseline_auc:.4f}",
)

print(
    "95% CI:",
    (
        f"[{baseline_auc_ci[0]:.4f}, "
        f"{baseline_auc_ci[1]:.4f}]"
    ),
)

print(
    "Average Precision:",
    f"{baseline_ap:.4f}",
)

print(
    "Frozen threshold:",
    f"{baseline_threshold:.3f}",
)

print(
    "Balanced Accuracy:",
    f"{baseline_metrics.balanced_accuracy:.4f}",
)

print(
    "Precision:",
    f"{baseline_metrics.precision:.4f}",
)

print(
    "Recall:",
    f"{baseline_metrics.recall:.4f}",
)

print(
    "Specificity:",
    f"{baseline_metrics.specificity:.4f}",
)

print(
    "F1:",
    f"{baseline_metrics.f1:.4f}",
)


print()
print("CANDIDATE — BGE Top-5")
print("-" * 80)

print(
    "ROC-AUC:",
    f"{candidate_auc:.4f}",
)

print(
    "95% CI:",
    (
        f"[{candidate_auc_ci[0]:.4f}, "
        f"{candidate_auc_ci[1]:.4f}]"
    ),
)

print(
    "Average Precision:",
    f"{candidate_ap:.4f}",
)

print(
    "Frozen threshold:",
    f"{candidate_threshold:.3f}",
)

print(
    "Balanced Accuracy:",
    f"{candidate_metrics.balanced_accuracy:.4f}",
)

print(
    "Precision:",
    f"{candidate_metrics.precision:.4f}",
)

print(
    "Recall:",
    f"{candidate_metrics.recall:.4f}",
)

print(
    "Specificity:",
    f"{candidate_metrics.specificity:.4f}",
)

print(
    "F1:",
    f"{candidate_metrics.f1:.4f}",
)


print()
print("RETRIEVAL")
print("-" * 80)

print(
    "Baseline AnyGold:",
    f"{baseline_any_gold / gold_eligible:.4f}",
)

print(
    "Baseline AllGold:",
    f"{baseline_all_gold / gold_eligible:.4f}",
)

print(
    "Candidate AnyGold:",
    f"{candidate_any_gold / gold_eligible:.4f}",
)

print(
    "Candidate AllGold:",
    f"{candidate_all_gold / gold_eligible:.4f}",
)


print()
print("COMPARISON")
print("-" * 80)

print(
    "AUC delta:",
    f"{auc_delta:+.4f}",
)

print(
    "Delta 95% CI:",
    (
        f"[{delta_auc_ci[0]:+.4f}, "
        f"{delta_auc_ci[1]:+.4f}]"
    ),
)

print(
    "Status:",
    generalization_status,
)


print()
print(
    "Threshold tuned on TEST: NO"
)

print(
    "Architecture selected on TEST: NO"
)

print(
    "RRF challenger evaluated: NO"
)

print()
print(
    "Saved:",
    RESULT_PATH,
)

print("=" * 80)
