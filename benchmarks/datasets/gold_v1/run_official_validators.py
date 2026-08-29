"""
Run the project's OFFICIAL validators against samples_all.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation_schema import (
    Adjudication,
    FailureLabel,
    ReviewerAnnotation,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)
from benchmarks.validation_coverage import (
    CoverageRequirements,
    calculate_validation_coverage,
    validate_benchmark_coverage,
)
from benchmarks.gold_validation import validate_gold_dataset
from benchmarks.validation_split import validate_no_split_leakage

BASE_DIR = Path(__file__).parent
SAMPLES_PATH = BASE_DIR / "samples_all.json"


def build_sample(raw: dict) -> ValidationSample:
    reviewers = tuple(
        ReviewerAnnotation(
            reviewer_id=r["reviewer_id"],
            label=FailureLabel(r["label"]),
            notes=r.get("notes"),
        )
        for r in raw.get("reviewers", [])
    )

    adjudication = None
    if raw.get("adjudication"):
        a = raw["adjudication"]
        adjudication = Adjudication(
            adjudicator_id=a["adjudicator_id"],
            label=FailureLabel(a["label"]),
            notes=a.get("notes"),
        )

    return ValidationSample(
        sample_id=raw["sample_id"],
        split=ValidationSplit(raw["split"]),
        gold_label=FailureLabel(raw["gold_label"]),
        language=SupportedLanguage(raw["language"]),
        domain=raw["domain"],
        question=raw["question"],
        answer=raw["answer"],
        contexts=tuple(raw["contexts"]),
        model_provider=raw["model_provider"],
        model_name=raw["model_name"],
        retriever_name=raw["retriever_name"],
        reference_answer=raw.get("reference_answer"),
        prompt=raw.get("prompt"),
        reviewers=reviewers,
        adjudication=adjudication,
        source_fact_id=raw.get("source_fact_id"),
    )


def main() -> None:
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples = [build_sample(r) for r in raw_samples]
    print(f"Loaded {len(samples)} samples as ValidationSample objects.\n")

    print("=== Coverage Check ===")
    try:
        coverage = validate_benchmark_coverage(
            samples,
            requirements=CoverageRequirements(
                minimum_samples_per_label=10,
                minimum_languages=2,
                minimum_domains=3,
                minimum_models=2,
                minimum_retrievers=2,
            ),
        )
        print("PASSED coverage validation.\n")
        print(f"Total samples: {coverage.sample_count}")
        print(f"By label: {coverage.by_label}")
        print(f"By language: {coverage.by_language}")
        print(f"By domain: {coverage.by_domain}")
        print(f"By model: {coverage.by_model}")
        print(f"By retriever: {coverage.by_retriever}")
        print(f"By split: {coverage.by_split}")
    except ValueError as e:
        print(f"FAILED coverage validation: {e}")

    print("\n=== Gold Label Validation ===")
    try:
        validate_gold_dataset(samples)
        print("PASSED gold label validation.")
    except ValueError as e:
        print(f"FAILED gold label validation: {e}")

    print("\n=== Split Leakage Validation ===")
    development = [
        sample
        for sample in samples
        if sample.split == ValidationSplit.DEVELOPMENT
    ]
    held_out = [
        sample
        for sample in samples
        if sample.split == ValidationSplit.HELD_OUT
    ]

    try:
        validate_no_split_leakage(
            development,
            held_out,
        )
        print(
            "PASSED split leakage validation "
            f"(development={len(development)}, "
            f"held_out={len(held_out)})."
        )
    except ValueError as e:
        print(f"FAILED split leakage validation: {e}")


if __name__ == "__main__":
    main()
