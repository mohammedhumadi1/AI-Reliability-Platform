"""Apply Reviewer B and compute direct Reviewer A vs Reviewer B agreement."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from reviewer_a_annotations import REVIEWER_A_JUDGMENTS
from reviewer_b_annotations import REVIEWER_B_ID, REVIEWER_B_JUDGMENTS

BASE_DIR = Path(__file__).parent
MAPPING_PATH = BASE_DIR / "internal_id_mapping.json"
SAMPLES_PATH = BASE_DIR / "samples_all.json"
REPORT_PATH = BASE_DIR / "reviewer_ab_agreement_report.json"

REVIEWER_A_ID = "sara_reviewer"


def cohen_kappa(a_labels: list[str], b_labels: list[str]) -> float:
    if len(a_labels) != len(b_labels) or not a_labels:
        raise ValueError("Reviewer label lists must have the same non-zero length.")

    total = len(a_labels)
    observed = sum(a == b for a, b in zip(a_labels, b_labels)) / total

    a_counts = Counter(a_labels)
    b_counts = Counter(b_labels)
    labels = set(a_counts) | set(b_counts)

    expected = sum(
        (a_counts[label] / total) * (b_counts[label] / total)
        for label in labels
    )

    if expected == 1.0:
        return 1.0

    return (observed - expected) / (1.0 - expected)


def main() -> None:
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)

    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    candidate_to_sample = {
        row["neutral_id"]: row["real_sample_id"]
        for row in mapping
    }

    if len(candidate_to_sample) != 50:
        raise ValueError(f"Expected 50 mappings, found {len(candidate_to_sample)}.")

    if set(REVIEWER_A_JUDGMENTS) != set(REVIEWER_B_JUDGMENTS):
        raise ValueError("Reviewer A and Reviewer B candidate sets do not match.")

    if set(REVIEWER_A_JUDGMENTS) != set(candidate_to_sample):
        raise ValueError("Reviewer judgments and internal mapping candidate sets do not match.")

    sample_by_id = {sample["sample_id"]: sample for sample in samples}

    a_labels: list[str] = []
    b_labels: list[str] = []
    agreements = []
    disagreements = []

    for candidate_id in sorted(REVIEWER_A_JUDGMENTS):
        sample_id = candidate_to_sample[candidate_id]
        sample = sample_by_id[sample_id]

        a_label = REVIEWER_A_JUDGMENTS[candidate_id]
        b_label = REVIEWER_B_JUDGMENTS[candidate_id]

        a_labels.append(a_label)
        b_labels.append(b_label)

        existing_reviewers = {
            reviewer["reviewer_id"]: reviewer
            for reviewer in sample.get("reviewers", [])
        }

        existing_a = existing_reviewers.get(REVIEWER_A_ID)
        if existing_a is None:
            raise ValueError(
                f"{sample_id} is missing Reviewer A annotation."
            )

        if existing_a["label"] != a_label:
            raise ValueError(
                f"{sample_id}: stored Reviewer A label "
                f"{existing_a['label']} != expected {a_label}."
            )

        existing_reviewers[REVIEWER_B_ID] = {
            "reviewer_id": REVIEWER_B_ID,
            "label": b_label,
            "notes": None,
        }

        sample["reviewers"] = [
            existing_reviewers[REVIEWER_A_ID],
            existing_reviewers[REVIEWER_B_ID],
        ]

        if a_label == b_label:
            agreements.append(sample_id)
        else:
            disagreements.append(
                {
                    "candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "reviewer_a_label": a_label,
                    "reviewer_b_label": b_label,
                }
            )

    total = len(a_labels)
    agreement_rate = len(agreements) / total
    kappa = cohen_kappa(a_labels, b_labels)

    report = {
        "reviewer_a_id": REVIEWER_A_ID,
        "reviewer_b_id": REVIEWER_B_ID,
        "total_samples": total,
        "agreements": len(agreements),
        "disagreements": len(disagreements),
        "agreement_rate": round(agreement_rate, 4),
        "cohen_kappa": round(kappa, 4),
        "reviewer_a_distribution": dict(sorted(Counter(a_labels).items())),
        "reviewer_b_distribution": dict(sorted(Counter(b_labels).items())),
        "disagreement_details": disagreements,
    }

    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"A vs B Agreement: {len(agreements)}/{total} ({agreement_rate:.1%})")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Disagreements: {len(disagreements)}")
    print()
    for item in disagreements:
        print(
            f"{item['candidate_id']} | {item['sample_id']} | "
            f"A={item['reviewer_a_label']} | B={item['reviewer_b_label']}"
        )


if __name__ == "__main__":
    main()
