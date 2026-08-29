"""
Build the final 10 RETRIEVAL_FAILURE samples using genuine indexing-gap
misses (genuine_retrieval_misses.json): the correct info is verified
absent (both languages) from the retrieved contexts. Run a real LLM
against each selected case's actual wrong retrieved context, so the
answer reflects what the model genuinely said given only that (wrong)
context.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)

BASE_DIR = Path(__file__).parent
MISSES_PATH = BASE_DIR / "genuine_retrieval_misses.json"
OUTPUT_PATH = BASE_DIR / "samples_retrieval_failure.json"

MODELS_CYCLE = ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

# Select 10 of the 24 genuine misses, spread across domains/languages.
SELECTION = [
    ("it_support", 0, "en"),
    ("it_support", 1, "ar"),
    ("it_support", 2, "en"),
    ("it_support", 3, "ar"),
    ("travel_expense", 0, "en"),
    ("travel_expense", 1, "ar"),
    ("travel_expense", 2, "en"),
    ("customer_returns", 1, "ar"),
    ("customer_returns", 2, "en"),
    ("customer_returns", 3, "ar"),
]


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(model_name: str, context: str, question: str) -> str:
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the information in the context "
        "above. Answer directly and concisely in one sentence."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    return strip_thinking(raw)


def main() -> None:
    with open(MISSES_PATH, encoding="utf-8") as f:
        misses = json.load(f)

    lookup = {
        (m["domain"], m["fact_index"], m["language"]): m for m in misses
    }

    samples = []
    for i, (domain, fact_index, lang) in enumerate(SELECTION):
        miss = lookup[(domain, fact_index, lang)]
        # The "wrong" context shown to the model: the actually retrieved
        # (genuinely wrong) chunks — join them as the RAG system would.
        wrong_context = miss["retrieved_contexts"][0]

        model_name = MODELS_CYCLE[i % 2]
        print(f"[{domain}/{lang}/fact_{fact_index}] asking {model_name}...")
        answer = ask_model(model_name, wrong_context, miss["question"])
        time.sleep(1)

        sample_id = (
            f"gold-{domain.replace('_', '-')}-{lang}-"
            f"{fact_index:03d}-retrieval-failure"
        )

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "RETRIEVAL_FAILURE",
            "language": lang,
            "domain": domain,
            "question": miss["question"],
            "answer": answer,
            "contexts": miss["retrieved_contexts"],
            "model_provider": "groq",
            "model_name": model_name,
            "retriever_name": "chroma_minilm_topk3",
            "reference_answer": miss["reference_answer"],
            "prompt": None,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt {len(samples)} FINAL RETRIEVAL_FAILURE samples (genuine misses).")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
