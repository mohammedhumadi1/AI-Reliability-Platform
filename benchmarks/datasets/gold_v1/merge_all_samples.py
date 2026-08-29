"""
Merge all 5 label files into one complete samples.json (50 total).
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "samples_all.json"

FILES = [
    "samples_healthy.json",
    "samples_retrieval_failure.json",
    "samples_generation_failure.json",
    "samples_knowledge_base_failure.json",
    "samples_prompt_failure.json",
]


def main() -> None:
    all_samples = []
    for filename in FILES:
        path = BASE_DIR / filename
        with open(path, encoding="utf-8") as f:
            samples = json.load(f)
        all_samples.extend(samples)
        print(f"{filename}: {len(samples)} samples")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\nTotal merged: {len(all_samples)} samples")
    print(f"Saved to {OUTPUT_PATH}")

    ids = [s["sample_id"] for s in all_samples]
    if len(ids) != len(set(ids)):
        print("\n⚠️  WARNING: duplicate sample_id detected!")
    else:
        print("All sample_id values are unique. ✅")


if __name__ == "__main__":
    main()
