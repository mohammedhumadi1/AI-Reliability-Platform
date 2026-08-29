"""
Build 10 GENERATION_FAILURE candidate samples: real correct context,
paired with a real model answer that genuinely came from a different
question, so it contradicts this context.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESPONSES_PATH = BASE_DIR / "model_responses.json"
OUTPUT_PATH = BASE_DIR / "samples_generation_failure.json"

GENERATION_FAILURE_SELECTION = [
    ("it_support", 0, "en", 2),
    ("it_support", 1, "ar", 3),
    ("it_support", 2, "en", 0),
    ("travel_expense", 0, "ar", 2),
    ("travel_expense", 1, "en", 3),
    ("travel_expense", 2, "ar", 0),
    ("customer_returns", 0, "en", 3),
    ("customer_returns", 1, "ar", 2),
    ("customer_returns", 2, "en", 0),
    ("customer_returns", 3, "ar", 1),
]


def main() -> None:
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)

    lookup = {
        (r["domain"], r["fact_index"], r["language"]): r for r in responses
    }

    samples = []

    for domain, fact_index, lang, wrong_fact_index in GENERATION_FAILURE_SELECTION:
        question_info = lookup[(domain, fact_index, lang)]
        wrong_answer_info = lookup[(domain, wrong_fact_index, lang)]

        sample_id = f"gold-{domain.replace('_', '-')}-{lang}-{fact_index:03d}-generation-failure"

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "GENERATION_FAILURE",
            "language": lang,
            "domain": domain,
            "question": question_info["question"],
            "answer": wrong_answer_info["model_answer"],
            "contexts": [question_info["context"]],
            "model_provider": wrong_answer_info["model_provider"],
            "model_name": wrong_answer_info["model_name"],
            "retriever_name": "chroma_minilm_topk1",
            "reference_answer": question_info["reference_answer"],
            "prompt": None,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Built {len(samples)} GENERATION_FAILURE candidate samples.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
