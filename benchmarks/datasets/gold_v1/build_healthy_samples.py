"""
Build 10 HEALTHY candidate samples (5 English + 5 Arabic), using real
model_provider/model_name/answer and real retriever_name/context.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESPONSES_PATH = BASE_DIR / "model_responses.json"
RETRIEVER_PATH = BASE_DIR / "retriever_results.json"
OUTPUT_PATH = BASE_DIR / "samples_healthy.json"

RETRIEVER_NAME_FOR_HEALTHY = "chroma_minilm_topk1"

HEALTHY_SELECTION = [
    ("it_support", 0, "en"),
    ("it_support", 1, "ar"),
    ("it_support", 2, "en"),
    ("it_support", 3, "ar"),
    ("travel_expense", 0, "en"),
    ("travel_expense", 1, "ar"),
    ("travel_expense", 2, "en"),
    ("customer_returns", 3, "ar"),
    ("customer_returns", 1, "en"),
    ("customer_returns", 2, "ar"),
]


def main() -> None:
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)
    with open(RETRIEVER_PATH, encoding="utf-8") as f:
        retriever_results = json.load(f)

    response_index = {
        (r["domain"], r["fact_index"], r["language"], r["model_name"]): r
        for r in responses
    }
    retriever_index = {
        (r["domain"], r["fact_index"], r["language"], r["retriever_name"]): r
        for r in retriever_results
    }

    models_cycle = ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

    samples = []
    for i, (domain, fact_index, lang) in enumerate(HEALTHY_SELECTION):
        model_name = models_cycle[i % 2]
        r = response_index[(domain, fact_index, lang, model_name)]

        retrieval = retriever_index[(domain, fact_index, lang, RETRIEVER_NAME_FOR_HEALTHY)]
        retrieved_context = retrieval["retrieved_contexts"][0]

        sample_id = f"gold-{domain.replace('_', '-')}-{lang}-{fact_index:03d}-healthy"

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "HEALTHY",
            "language": lang,
            "domain": domain,
            "question": r["question"],
            "answer": r["model_answer"],
            "contexts": [retrieved_context],
            "model_provider": r["model_provider"],
            "model_name": r["model_name"],
            "retriever_name": RETRIEVER_NAME_FOR_HEALTHY,
            "reference_answer": r["reference_answer"],
            "prompt": None,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Built {len(samples)} HEALTHY candidate samples.")
    print(f"Saved to {OUTPUT_PATH}")
    print("All fields now use real model_provider/model_name/retriever_name.")


if __name__ == "__main__":
    main()
