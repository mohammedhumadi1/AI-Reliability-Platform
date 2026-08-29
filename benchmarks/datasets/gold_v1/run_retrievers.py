"""
Set up a real retriever using the project's existing Chroma-based
knowledge_base/vector_store.py, isolated under its own project_id.
Runs TWO real retriever configurations (top_k=1 and top_k=3).
"""
from __future__ import annotations

import json
from pathlib import Path

from knowledge_base.vector_store import add_chunks, query_similar_chunks

BASE_DIR = Path(__file__).parent
FACTS_PATH = BASE_DIR / "source_facts.json"
QUESTIONS_PATH = BASE_DIR / "questions.json"
OUTPUT_PATH = BASE_DIR / "retriever_results.json"

PROJECT_ID = "gold_benchmark_v1"

RETRIEVER_CONFIGS = [
    {"retriever_name": "chroma_minilm_topk1", "top_k": 1},
    {"retriever_name": "chroma_minilm_topk3", "top_k": 3},
]


def index_facts() -> None:
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts_by_domain = json.load(f)

    for domain, langs in facts_by_domain.items():
        for lang, facts in langs.items():
            add_chunks(
                project_id=PROJECT_ID,
                chunks=facts,
                source=f"{domain}_{lang}_source_facts",
                document_id=f"{domain}_{lang}",
            )
    print(f"Indexed source facts into isolated project '{PROJECT_ID}'.")


def main() -> None:
    index_facts()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions_by_domain = json.load(f)

    results = []

    for domain, items in questions_by_domain.items():
        for item in items:
            fact_index = item["fact_index"]
            for lang in ("en", "ar"):
                question = item[lang]["question"]

                for config in RETRIEVER_CONFIGS:
                    matches = query_similar_chunks(
                        project_id=PROJECT_ID,
                        query=question,
                        top_k=config["top_k"],
                    )
                    retrieved_texts = [m["text"] for m in matches]

                    results.append({
                        "domain": domain,
                        "fact_index": fact_index,
                        "language": lang,
                        "question": question,
                        "retriever_name": config["retriever_name"],
                        "top_k": config["top_k"],
                        "retrieved_contexts": retrieved_texts,
                    })
                    print(f"[{domain}/{lang}/fact_{fact_index}] {config['retriever_name']} -> {len(retrieved_texts)} chunk(s)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
