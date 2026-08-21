import hashlib
import json
import random
from pathlib import Path


DATASET_NAME = "galileo-ai/ragbench"
BENCHMARK_VERSION = "ragbench-validation-v1"
SPLIT = "validation"
BASE_SEED = 20260821

SUBSETS = [
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

SAMPLES_PER_SUBSET = 50
SAMPLES_PER_CLASS = 25

OUTPUT_PATH = Path(
    "benchmarks/manifests/"
    "ragbench_validation_v1.json"
)


def stable_subset_seed(
    subset: str,
) -> int:
    value = (
        f"{BASE_SEED}:{subset}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        value
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def select_balanced_indices(
    labels: list[bool],
    samples_per_class: int,
    seed: int,
) -> list[int]:
    if samples_per_class <= 0:
        raise ValueError(
            "samples_per_class must be "
            "greater than zero."
        )

    supported = [
        index
        for index, label in enumerate(labels)
        if label
    ]

    unsupported = [
        index
        for index, label in enumerate(labels)
        if not label
    ]

    if len(supported) < samples_per_class:
        raise ValueError(
            "Not enough supported samples."
        )

    if len(unsupported) < samples_per_class:
        raise ValueError(
            "Not enough unsupported samples."
        )

    rng = random.Random(seed)

    selected_supported = rng.sample(
        supported,
        samples_per_class,
    )

    selected_unsupported = rng.sample(
        unsupported,
        samples_per_class,
    )

    return sorted(
        selected_supported
        + selected_unsupported
    )


def build_manifest() -> dict:
    from datasets import load_dataset

    samples = []
    summary = {}

    for subset in SUBSETS:
        print(
            f"Loading {subset}..."
        )

        dataset = load_dataset(
            DATASET_NAME,
            subset,
            split=SPLIT,
        )

        labels = [
            bool(value)
            for value
            in dataset["adherence_score"]
        ]

        indices = (
            select_balanced_indices(
                labels=labels,
                samples_per_class=(
                    SAMPLES_PER_CLASS
                ),
                seed=stable_subset_seed(
                    subset
                ),
            )
        )

        subset_supported = 0
        subset_unsupported = 0

        for row_index in indices:
            row = dataset[row_index]

            supported = bool(
                row["adherence_score"]
            )

            if supported:
                subset_supported += 1
            else:
                subset_unsupported += 1

            samples.append(
                {
                    "subset": subset,
                    "split": SPLIT,
                    "row_index": row_index,
                    "sample_id": str(
                        row["id"]
                    ),
                    "supported": supported,
                }
            )

        assert (
            len(indices)
            == SAMPLES_PER_SUBSET
        )

        assert (
            subset_supported
            == SAMPLES_PER_CLASS
        )

        assert (
            subset_unsupported
            == SAMPLES_PER_CLASS
        )

        summary[subset] = {
            "total": len(indices),
            "supported": (
                subset_supported
            ),
            "unsupported": (
                subset_unsupported
            ),
        }

    supported_total = sum(
        int(sample["supported"])
        for sample in samples
    )

    unsupported_total = (
        len(samples)
        - supported_total
    )

    expected_total = (
        len(SUBSETS)
        * SAMPLES_PER_SUBSET
    )

    assert len(samples) == expected_total
    assert supported_total == 250
    assert unsupported_total == 250

    return {
        "benchmark_version": (
            BENCHMARK_VERSION
        ),
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "base_seed": BASE_SEED,
        "selection_strategy": (
            "Equal-domain balanced "
            "sampling: 25 supported and "
            "25 unsupported per subset."
        ),
        "subset_count": len(SUBSETS),
        "sample_count": len(samples),
        "supported_count": (
            supported_total
        ),
        "unsupported_count": (
            unsupported_total
        ),
        "subsets": SUBSETS,
        "summary": summary,
        "samples": samples,
    }


def main() -> None:
    manifest = build_manifest()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RAGBENCH VALIDATION V1 MANIFEST")
    print("=" * 72)

    for subset, stats in (
        manifest["summary"].items()
    ):
        print(
            f"{subset:<14}"
            f" total={stats['total']:>3}"
            f" supported="
            f"{stats['supported']:>2}"
            f" unsupported="
            f"{stats['unsupported']:>2}"
        )

    print("-" * 72)
    print(
        "Total:      ",
        manifest["sample_count"],
    )
    print(
        "Supported:  ",
        manifest["supported_count"],
    )
    print(
        "Unsupported:",
        manifest["unsupported_count"],
    )
    print(
        "Saved:      ",
        OUTPUT_PATH,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
