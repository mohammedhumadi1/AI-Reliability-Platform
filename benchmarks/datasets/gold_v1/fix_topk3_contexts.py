"""
Fix RETRIEVAL_FAILURE samples per Mohammed's review: the retriever_name
says 'chroma_minilm_topk3' but only ONE context was shown. Since top_k=3
genuinely retrieves 3 chunks, all 3 real retrieved contexts should be
shown, to accurately represent what the retriever actually returned.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RETRIEVER_PATH = BASE_DIR / "retriever_results.json"
SAMPLES_PATH = BASE_DIR / "samples_retrieval_failure.json"

RETRIEVER_NAME = "chroma_minilm_topk3"


def main() -> None:
    with open(RETRIEVER_PATH, encoding="utf-8") as f:
        retriever_results = json.load(f)
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    retriever_index = {
        (r["domain"], r["fact_index"], r["language"], r["retriever_name"]): r
        for r in retriever_results
    }

    updated = 0
    for sample in samples:
        domain = sample["domain"]
        lang = sample["language"]
        parts = sample["sample_id"].split("-")
        fact_index = int(parts[-3])

        retrieval = retriever_index[(domain, fact_index, lang, RETRIEVER_NAME)]
        all_three_contexts = retrieval["retrieved_contexts"]

        sample["contexts"] = all_three_contexts
        updated += 1

    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Updated {updated} RETRIEVAL_FAILURE samples to show all real top_k=3 contexts.")
    print(f"Saved to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
