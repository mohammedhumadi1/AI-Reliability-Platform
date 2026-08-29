"""
Simulate a real "indexing gap" retrieval failure, fixed version:
remove BOTH language versions of a domain's real facts together (not
just one language), so the underlying information is genuinely absent
in ANY language — this matches Mohammed's requirement precisely,
avoiding the earlier flaw where a translated version of the same fact
remained retrievable.
"""
from __future__ import annotations

import json
from pathlib import Path

from knowledge_base.vector_store import (
    add_chunks,
    delete_document,
    query_similar_chunks,
)

BASE_DIR = Path(__file__).parent
FACTS_PATH = BASE_DIR / "source_facts.json"
RESPONSES_PATH = BASE_DIR / "model_responses.json"
OUTPUT_PATH = BASE_DIR / "genuine_retrieval_misses.json"

PROJECT_ID = "gold_benchmark_v1"
TOP_K = 3

DOMAINS_TO_TEST = ["it_support", "travel_expense", "customer_returns"]


def main() -> None:
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts_by_domain = json.load(f)
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)

    seen = set()
    questions = []
    for r in responses:
        key = (r["domain"], r["fact_index"], r["language"])
        if key in seen:
            continue
        seen.add(key)
        questions.append(r)

    genuine_misses = []

    for domain in DOMAINS_TO_TEST:
        doc_id_en = f"{domain}_en"
        doc_id_ar = f"{domain}_ar"

        deleted_en = delete_document(PROJECT_ID, doc_id_en)
        deleted_ar = delete_document(PROJECT_ID, doc_id_ar)
        print(
            f"\n--- Removed real facts (BOTH languages) for {domain} "
            f"(en={deleted_en}, ar={deleted_ar}) ---"
        )

        group_questions = [q for q in questions if q["domain"] == domain]

        for q in group_questions:
            matches = query_similar_chunks(
                project_id=PROJECT_ID,
                query=q["question"],
                top_k=TOP_K,
            )
            retrieved_texts = [m["text"] for m in matches]

            # Sanity check: the correct fact must be absent in BOTH
            # languages, not just the query's own language.
            correct_en = facts_by_domain[domain]["en"][q["fact_index"]]
            correct_ar = facts_by_domain[domain]["ar"][q["fact_index"]]

            assert correct_en not in retrieved_texts, (
                f"Unexpected: EN correct context leaked for "
                f"{domain}/fact_{q['fact_index']}"
            )
            assert correct_ar not in retrieved_texts, (
                f"Unexpected: AR correct context leaked for "
                f"{domain}/fact_{q['fact_index']}"
            )

            print(
                f"  [{domain}/{q['language']}/fact_{q['fact_index']}] "
                f"genuine miss confirmed (both languages absent), "
                f"{len(retrieved_texts)} chunk(s) retrieved"
            )

            genuine_misses.append(
                {
                    "domain": q["domain"],
                    "fact_index": q["fact_index"],
                    "language": q["language"],
                    "question": q["question"],
                    "correct_context": q["context"],
                    "reference_answer": q["reference_answer"],
                    "retrieved_contexts": retrieved_texts,
                }
            )

        # Restore both languages for this domain.
        add_chunks(
            project_id=PROJECT_ID,
            chunks=facts_by_domain[domain]["en"],
            source=f"{domain}_en_source_facts",
            document_id=doc_id_en,
        )
        add_chunks(
            project_id=PROJECT_ID,
            chunks=facts_by_domain[domain]["ar"],
            source=f"{domain}_ar_source_facts",
            document_id=doc_id_ar,
        )
        print(f"--- Restored real facts (both languages) for {domain} ---")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(genuine_misses, f, ensure_ascii=False, indent=2)

    print(f"\nTotal genuine retrieval misses found: {len(genuine_misses)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
