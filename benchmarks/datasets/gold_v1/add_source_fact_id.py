"""
Add a `source_fact_id` field to every sample in each of the 5 label
files, per Mohammed's review: this groups samples by their underlying
source fact (domain + fact_index), regardless of language or label, so
a future split can ensure the SAME fact never appears in both
development and held-out (avoiding leakage).

This does NOT perform the split — Mohammed explicitly said not to run
the 70/30 split yet. This only adds the grouping key needed for it.

For KNOWLEDGE_BASE_FAILURE samples (which are about info NOT in the
source facts), source_fact_id is set to a domain-level placeholder like
"customer_returns_OUT_OF_SCOPE" since there's no single fact_index they
map to.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

FILES = [
    "samples_healthy.json",
    "samples_retrieval_failure.json",
    "samples_generation_failure.json",
    "samples_knowledge_base_failure.json",
    "samples_prompt_failure.json",
]

# sample_id pattern: gold-{domain-with-dashes}-{lang}-{fact_index}-{label-suffix}
SAMPLE_ID_RE = re.compile(
    r"^gold-(?P<domain>[a-z\-]+)-(?P<lang>en|ar)-(?P<fact_index>\d+)-"
)


def derive_source_fact_id(sample: dict) -> str:
    sample_id = sample["sample_id"]
    domain = sample["domain"]

    if sample["gold_label"] == "KNOWLEDGE_BASE_FAILURE":
        return f"{domain}_OUT_OF_SCOPE_{sample_id}"

    match = SAMPLE_ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Could not parse sample_id: {sample_id}")

    fact_index = match.group("fact_index")
    return f"{domain}_{fact_index}"


def main() -> None:
    for filename in FILES:
        path = BASE_DIR / filename
        with open(path, encoding="utf-8") as f:
            samples = json.load(f)

        for sample in samples:
            sample["source_fact_id"] = derive_source_fact_id(sample)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"{filename}: added source_fact_id to {len(samples)} samples.")

    print("\nDone. Re-run merge_all_samples.py to regenerate samples_all.json.")


if __name__ == "__main__":
    main()
