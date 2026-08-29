"""Apply deterministic resolution to Reviewer A/B disagreements."""

from __future__ import annotations

import json
from pathlib import Path

from disagreement_resolution import (
    DISAGREEMENT_RESOLUTIONS,
    RESOLVER_ID,
)

BASE_DIR = Path(__file__).parent
SAMPLES_PATH = BASE_DIR / "samples_all.json"
AGREEMENT_PATH = BASE_DIR / "reviewer_ab_agreement_report.json"
OUTPUT_REPORT = BASE_DIR / "disagreement_resolution_report.json"

EXPECTED_REVIEWERS = {"sara_reviewer", "mohammed_reviewer"}


def main() -> None:
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    with open(AGREEMENT_PATH, encoding="utf-8") as f:
        agreement = json.load(f)

    disagreements = agreement["disagreement_details"]

    if agreement["total_samples"] != 50:
        raise ValueError("Expected exactly 50 reviewed samples.")

    if len(disagreements) != 12:
        raise ValueError(
            f"Expected exactly 12 disagreements, found {len(disagreements)}."
        )

    disagreement_ids = {
        item["candidate_id"]
        for item in disagreements
    }

    resolution_ids = set(DISAGREEMENT_RESOLUTIONS)

    if disagreement_ids != resolution_ids:
        missing = sorted(disagreement_ids - resolution_ids)
        extra = sorted(resolution_ids - disagreement_ids)
        raise ValueError(
            f"Resolution mapping mismatch. Missing={missing}, extra={extra}"
        )

    allowed_pair = {
        "RETRIEVAL_FAILURE",
        "KNOWLEDGE_BASE_FAILURE",
    }

    for item in disagreements:
        pair = {
            item["reviewer_a_label"],
            item["reviewer_b_label"],
        }
        if pair != allowed_pair:
            raise ValueError(
                f"{item['candidate_id']} is not a Retrieval-vs-KB disagreement: {pair}"
            )

    sample_by_id = {
        sample["sample_id"]: sample
        for sample in samples
    }

    resolved_details = []

    for item in disagreements:
        candidate_id = item["candidate_id"]
        sample_id = item["sample_id"]
        sample = sample_by_id[sample_id]

        reviewers = {
            r["reviewer_id"]: r["label"]
            for r in sample.get("reviewers", [])
        }

        if set(reviewers) != EXPECTED_REVIEWERS:
            raise ValueError(
                f"{sample_id} reviewers are {set(reviewers)}, "
                f"expected {EXPECTED_REVIEWERS}."
            )

        if reviewers["sara_reviewer"] != item["reviewer_a_label"]:
            raise ValueError(
                f"{sample_id}: Reviewer A label mismatch."
            )

        if reviewers["mohammed_reviewer"] != item["reviewer_b_label"]:
            raise ValueError(
                f"{sample_id}: Reviewer B label mismatch."
            )

        decision = DISAGREEMENT_RESOLUTIONS[candidate_id]

        sample["gold_label"] = decision["label"]
        sample["adjudication"] = {
            "adjudicator_id": RESOLVER_ID,
            "label": decision["label"],
            "notes": decision["reason"],
        }

        resolved_details.append(
            {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "reviewer_a_label": item["reviewer_a_label"],
                "reviewer_b_label": item["reviewer_b_label"],
                "resolved_label": decision["label"],
                "resolution_method": RESOLVER_ID,
                "reason": decision["reason"],
            }
        )

    # For agreed samples, freeze gold_label to reviewer consensus and
    # ensure no resolution metadata is present.
    disagreement_sample_ids = {
        item["sample_id"]
        for item in disagreements
    }

    for sample in samples:
        if sample["sample_id"] in disagreement_sample_ids:
            continue

        reviewers = sample.get("reviewers", [])

        if len(reviewers) != 2:
            raise ValueError(
                f"{sample['sample_id']} must have exactly two reviewers."
            )

        labels = {r["label"] for r in reviewers}

        if len(labels) != 1:
            raise ValueError(
                f"{sample['sample_id']} unexpectedly has reviewer disagreement."
            )

        consensus = next(iter(labels))
        sample["gold_label"] = consensus
        sample["adjudication"] = None

    report = {
        "resolution_method": RESOLVER_ID,
        "rule": {
            "RETRIEVAL_FAILURE": (
                "Required fact exists in authoritative source knowledge "
                "but is absent from retrieved contexts."
            ),
            "KNOWLEDGE_BASE_FAILURE": (
                "Required fact does not exist in authoritative source knowledge."
            ),
        },
        "total_samples": len(samples),
        "reviewer_agreements": agreement["agreements"],
        "reviewer_disagreements": len(disagreements),
        "resolved_disagreements": len(resolved_details),
        "resolved_details": resolved_details,
    }

    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Samples: {len(samples)}")
    print(f"Reviewer agreements: {agreement['agreements']}")
    print(f"Disagreements resolved: {len(resolved_details)}")
    print(f"Resolution method: {RESOLVER_ID}")

    counts = {}
    for sample in samples:
        label = sample["gold_label"]
        counts[label] = counts.get(label, 0) + 1

    print("Final gold-label counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
