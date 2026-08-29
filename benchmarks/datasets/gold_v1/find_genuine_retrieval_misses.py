"""
Index distractor facts alongside the real source facts (same isolated
Chroma project), then re-run retrieval (top_k=3) on all 24 questions
and identify GENUINE retrieval failures: cases where the correct
context does NOT appear anywhere in the top_k=3 results at all.

This is what Mohammed's review requires: a real retrieval failure means
the correct info was never retrieved, not just that we displayed a
wrong context alongside it.
"""
from __future__ import annotations

import json
from pathlib import Path

from knowledge_base.vector_store import add_chunks, query_similar_chunks

BASE_DIR = Path(__file__).parent
FACTS_PATH = BASE_DIR / "source_facts.json"
DISTRACTORS_PATH = BASE_DIR / "distractor_facts.json"
RESPONSES_PATH = BASE_DIR / "model_responses.json"
OUTPUT_PATH = BASE_DIR / "genuine_retrieval_misses.json"

PROJECT_ID = "gold_benchmark_v1"
TOP_K = 3


def index_distractors() -> None:
    with open(DISTRACTORS_PATH, encoding="utf-8") as f:
        distractors_by_domain = json.load(f)

    for domain, langs in distractors_by_domain.items():
        for lang, facts in langs.items():
            add_chunks(
                project_id=PROJECT_ID,
                chunks=facts,
                source=f"{domain}_{lang}_distractor_facts",
                document_id=f"{domain}_{lang}_distractors",
            )
    print("Indexed 36 distractor facts into the isolated project.")


def main() -> None:
    index_distractors()

    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)

    # One entry per (domain, fact_index, language) — dedupe from the
    # 48 model_responses entries (2 models each).
    seen = set()
    questions = []
    for r in responses:
        key = (r["domain"], r["fact_index"], r["language"])
        if key in seen:
            continue
        seen.add(key)
        questions.append(r)

    genuine_misses = []

    for q in questions:
        matches = query_similar_chunks(
            project_id=PROJECT_ID,
            query=q["question"],
            top_k=TOP_K,
        )
        retrieved_texts = [m["text"] for m in matches]

        correct_context = q["context"]
        is_genuine_miss = correct_context not in retrieved_texts

        status = "MISS (genuine failure)" if is_genuine_miss else "found in top_k"
        print(
            f"[{q['domain']}/{q['language']}/fact_{q['fact_index']}] {status}"
        )

        if is_genuine_miss:
            genuine_misses.append(
                {
                    "domain": q["domain"],
                    "fact_index": q["fact_index"],
                    "language": q["language"],
                    "question": q["question"],
                    "correct_context": correct_context,
                    "reference_answer": q["reference_answer"],
                    "retrieved_contexts": retrieved_texts,
                }
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(genuine_misses, f, ensure_ascii=False, indent=2)

    print(
        f"\nFound {len(genuine_misses)} genuine retrieval misses out of "
        f"{len(questions)} questions."
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
