"""
Build 10 RETRIEVAL_FAILURE candidate samples using real retrieval output
(top_k=3) paired with real model answers genuinely generated for that
wrong context.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESPONSES_PATH = BASE_DIR / "model_responses.json"
RETRIEVER_PATH = BASE_DIR / "retriever_results.json"
OUTPUT_PATH = BASE_DIR / "samples_retrieval_failure.json"

RETRIEVER_NAME = "chroma_minilm_topk3"

RETRIEVAL_FAILURE_SELECTION = [
    ("it_support", 0, "ar"),
    ("it_support", 1, "en"),
    ("it_support", 2, "ar"),
    ("it_support", 3, "en"),
    ("travel_expense", 0, "ar"),
    ("travel_expense", 1, "en"),
    ("travel_expense", 3, "ar"),
    ("customer_returns", 0, "en"),
    ("customer_returns", 1, "ar"),
    ("customer_returns", 2, "en"),
]


def main() -> None:
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)
    with open(RETRIEVER_PATH, encoding="utf-8") as f:
        retriever_results = json.load(f)

    retriever_index = {
        (r["domain"], r["fact_index"], r["language"], r["retriever_name"]): r
        for r in retriever_results
    }
    context_to_response = {}
    for r in responses:
        context_to_response.setdefault(r["context"], r)

    question_lookup = {
        (r["domain"], r["fact_index"], r["language"]): r for r in responses
    }

    samples = []
    skipped = []

    for domain, fact_index, lang in RETRIEVAL_FAILURE_SELECTION:
        q_info = question_lookup[(domain, fact_index, lang)]
        question = q_info["question"]
        correct_context = q_info["context"]

        retrieval = retriever_index[(domain, fact_index, lang, RETRIEVER_NAME)]
        retrieved = retrieval["retrieved_contexts"]

        wrong_context = next((c for c in retrieved if c != correct_context), None)

        if wrong_context is None:
            skipped.append((domain, fact_index, lang))
            continue

        wrong_context_response = context_to_response.get(wrong_context)
        if wrong_context_response is None:
            skipped.append((domain, fact_index, lang))
            continue

        sample_id = f"gold-{domain.replace('_', '-')}-{lang}-{fact_index:03d}-retrieval-failure"

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "RETRIEVAL_FAILURE",
            "language": lang,
            "domain": domain,
            "question": question,
            "answer": wrong_context_response["model_answer"],
            "contexts": [wrong_context],
            "model_provider": wrong_context_response["model_provider"],
            "model_name": wrong_context_response["model_name"],
            "retriever_name": RETRIEVER_NAME,
            "reference_answer": q_info["reference_answer"],
            "prompt": None,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Built {len(samples)} RETRIEVAL_FAILURE candidate samples.")
    print(f"Saved to {OUTPUT_PATH}")
    if skipped:
        print(f"\nSkipped {len(skipped)} (no wrong context found): {skipped}")


if __name__ == "__main__":
    main()
