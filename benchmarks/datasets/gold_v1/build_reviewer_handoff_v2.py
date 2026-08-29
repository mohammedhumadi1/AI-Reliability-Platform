"""
Rebuild the Reviewer B handoff file to be properly blinded, per
Mohammed's review.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
SAMPLES_PATH = BASE_DIR / "samples_all.json"
CANDIDATES_PATH = BASE_DIR / "reviewer_candidates.json"
MAPPING_PATH = BASE_DIR / "internal_id_mapping.json"

SHUFFLE_SEED = 20260825


def main() -> None:
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    rng = random.Random(SHUFFLE_SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    candidates = []
    mapping = []

    for i, s in enumerate(shuffled, start=1):
        neutral_id = f"candidate_{i:03d}"
        mapping.append({"neutral_id": neutral_id, "real_sample_id": s["sample_id"]})

        candidate = {
            "candidate_id": neutral_id,
            "question": s["question"],
            "contexts": s["contexts"],
            "answer": s["answer"],
            "prompt": s["prompt"],
        }
        candidates.append(candidate)

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Built {len(candidates)} properly blinded candidates.")
    print(f"Shuffle seed: {SHUFFLE_SEED}")
    print(f"-> {CANDIDATES_PATH}  (send this to Reviewer B)")
    print(f"-> {MAPPING_PATH}  (KEEP PRIVATE — do not send)")


if __name__ == "__main__":
    main()
