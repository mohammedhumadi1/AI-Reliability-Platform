"""Create the final deterministic 70/30 group-aware gold split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarks.datasets.gold_v1.run_official_validators import build_sample
from benchmarks.validation_split import (
    split_development_and_held_out,
    validate_no_split_leakage,
)

BASE_DIR = Path(__file__).parent
ALL_PATH = BASE_DIR / "samples_all.json"
DEV_PATH = BASE_DIR / "samples_development.json"
HELD_PATH = BASE_DIR / "samples_held_out.json"
REPORT_PATH = BASE_DIR / "split_report.json"

SEED = 20260824
DEVELOPMENT_FRACTION = 0.70


def _counts(samples):
    return dict(
        sorted(
            Counter(
                sample.gold_label.value
                for sample in samples
            ).items()
        )
    )


def main() -> None:
    with open(ALL_PATH, encoding="utf-8") as f:
        raw_samples = json.load(f)

    if len(raw_samples) != 50:
        raise ValueError(
            f"Expected 50 samples, found {len(raw_samples)}."
        )

    samples = [
        build_sample(raw)
        for raw in raw_samples
    ]

    development, held_out = split_development_and_held_out(
        samples,
        development_fraction=DEVELOPMENT_FRACTION,
        seed=SEED,
    )

    validate_no_split_leakage(
        development,
        held_out,
    )

    if len(development) != 35 or len(held_out) != 15:
        raise ValueError(
            "Expected a 35/15 split, got "
            f"{len(development)}/{len(held_out)}."
        )

    expected_dev = {
        "GENERATION_FAILURE": 7,
        "HEALTHY": 7,
        "KNOWLEDGE_BASE_FAILURE": 7,
        "PROMPT_FAILURE": 7,
        "RETRIEVAL_FAILURE": 7,
    }

    expected_held = {
        "GENERATION_FAILURE": 3,
        "HEALTHY": 3,
        "KNOWLEDGE_BASE_FAILURE": 3,
        "PROMPT_FAILURE": 3,
        "RETRIEVAL_FAILURE": 3,
    }

    if _counts(development) != expected_dev:
        raise ValueError(
            f"Unexpected development distribution: {_counts(development)}"
        )

    if _counts(held_out) != expected_held:
        raise ValueError(
            f"Unexpected held-out distribution: {_counts(held_out)}"
        )

    dev_ids = {
        sample.sample_id
        for sample in development
    }
    held_ids = {
        sample.sample_id
        for sample in held_out
    }

    raw_by_id = {
        raw["sample_id"]: raw
        for raw in raw_samples
    }

    for raw in raw_samples:
        sample_id = raw["sample_id"]

        if sample_id in dev_ids:
            raw["split"] = "development"
        elif sample_id in held_ids:
            raw["split"] = "held_out"
        else:
            raise ValueError(
                f"Unassigned sample: {sample_id}"
            )

    development_raw = [
        raw_by_id[sample.sample_id]
        for sample in development
    ]

    held_out_raw = [
        raw_by_id[sample.sample_id]
        for sample in held_out
    ]

    dev_facts = {
        sample.source_fact_id
        for sample in development
        if sample.source_fact_id
    }

    held_facts = {
        sample.source_fact_id
        for sample in held_out
        if sample.source_fact_id
    }

    overlap = sorted(dev_facts & held_facts)

    if overlap:
        raise ValueError(
            f"source_fact_id leakage detected: {overlap}"
        )

    report = {
        "seed": SEED,
        "development_fraction": DEVELOPMENT_FRACTION,
        "total_samples": len(raw_samples),
        "development_samples": len(development),
        "held_out_samples": len(held_out),
        "development_label_counts": _counts(development),
        "held_out_label_counts": _counts(held_out),
        "source_fact_id_overlap": overlap,
        "leakage_validation": "PASSED",
    }

    for path, data in (
        (ALL_PATH, raw_samples),
        (DEV_PATH, development_raw),
        (HELD_PATH, held_out_raw),
        (REPORT_PATH, report),
    ):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

    print("Final split saved successfully.")
    print(f"Development: {len(development)}")
    print(f"Held-out: {len(held_out)}")
    print("Development labels:", _counts(development))
    print("Held-out labels:", _counts(held_out))
    print("source_fact_id overlap:", overlap)
    print("Leakage validation: PASSED")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()
