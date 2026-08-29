"""
Rebuild RETRIEVAL_FAILURE samples correctly per Mohammed's review:
only accept a "wrong" context that maps to a genuinely DIFFERENT fact
(not just a translation of the same fact).
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
FACTS_PATH = BASE_DIR / "source_facts.json"
RESPONSES_PATH = BASE_DIR / "model_responses.json"
RETRIEVER_PATH = BASE_DIR / "retriever_results.json"
OUTPUT_PATH = BASE_DIR / "samples_retrieval_failure.json"

RETRIEVER_NAME = "chroma_minilm_topk3"

ALL_QUESTIONS = (
    [("it_support", i, lang) for i in range(4) for lang in ("en", "ar")]
    + [("travel_expense", i, lang) for i in range(4) for lang in ("en", "ar")]
    + [("customer_returns", i, lang) for i in range(4) for lang in ("en", "ar")]
)


def build_context_origin_map(facts_by_domain: dict) -> dict:
    origin = {}
    for domain, langs in facts_by_domain.items():
        for lang, facts in langs.items():
            for fact_index, text in enumerate(facts):
                origin[text] = (domain, fact_index)
    return origin


def main() -> None:
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts_by_domain = json.load(f)
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)
    with open(RETRIEVER_PATH, encoding="utf-8") as f:
        retriever_results = json.load(f)

    context_origin = build_context_origin_map(facts_by_domain)

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

    candidates = []

    for domain, fact_index, lang in ALL_QUESTIONS:
        q_info = question_lookup[(domain, fact_index, lang)]
        own_origin = (domain, fact_index)

        retrieval = retriever_index[(domain, fact_index, lang, RETRIEVER_NAME)]
        retrieved = retrieval["retrieved_contexts"]

        genuine_wrong = None
        for c in retrieved:
            c_origin = context_origin.get(c)
            if c_origin is not None and (
                c_origin[0] != own_origin[0] or c_origin[1] != own_origin[1]
            ):
                genuine_wrong = c
                break

        if genuine_wrong is None:
            continue

        wrong_resp = context_to_response.get(genuine_wrong)
        if wrong_resp is None:
            continue

        candidates.append((domain, fact_index, lang, genuine_wrong, wrong_resp))

    print(f"Found {len(candidates)} genuine cross-fact retrieval-failure candidates out of 24 questions.")

    selected = candidates[:10]

    samples = []
    for domain, fact_index, lang, wrong_context, wrong_resp in selected:
        q_info = question_lookup[(domain, fact_index, lang)]

        sample_id = f"gold-{domain.replace('_', '-')}-{lang}-{fact_index:03d}-retrieval-failure"

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "RETRIEVAL_FAILURE",
            "language": lang,
            "domain": domain,
            "question": q_info["question"],
            "answer": wrong_resp["model_answer"],
            "contexts": [wrong_context],
            "model_provider": wrong_resp["model_provider"],
            "model_name": wrong_resp["model_name"],
            "retriever_name": RETRIEVER_NAME,
            "reference_answer": q_info["reference_answer"],
            "prompt": None,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt {len(samples)} corrected RETRIEVAL_FAILURE samples.")
    print(f"Saved to {OUTPUT_PATH}")
    if len(samples) < 10:
        print(f"\n⚠️  Only {len(samples)}/10 genuine candidates found. May need wider top_k.")


if __name__ == "__main__":
    main()
