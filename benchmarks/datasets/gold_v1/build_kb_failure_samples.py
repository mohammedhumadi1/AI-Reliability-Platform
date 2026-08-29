"""
Build 10 KNOWLEDGE_BASE_FAILURE candidate samples using real retriever
+ real LLM on out-of-scope questions.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from knowledge_base.vector_store import query_similar_chunks

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "kb_failure_questions.json"
OUTPUT_PATH = BASE_DIR / "samples_knowledge_base_failure.json"

PROJECT_ID = "gold_benchmark_v1"
RETRIEVER_NAME = "chroma_minilm_topk1"

MODELS = [
    {"provider": "groq", "name": "openai/gpt-oss-20b"},
    {"provider": "groq", "name": "qwen/qwen3.6-27b"},
]


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(model_name: str, context: str, question: str) -> str:
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the information in the context "
        "above. If the context does not contain the answer, say so "
        "explicitly. Answer directly and concisely in one sentence."
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
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    models_cycle = ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
    model_lookup = {m["name"]: m for m in MODELS}

    samples = []

    for i, item in enumerate(questions):
        domain = item["domain"]
        lang = item["language"]
        question = item["question"]

        matches = query_similar_chunks(project_id=PROJECT_ID, query=question, top_k=1)
        retrieved_context = matches[0]["text"] if matches else ""

        model_name = models_cycle[i % 2]
        model_info = model_lookup[model_name]

        print(f"[{domain}/{lang}] asking {model_name}: {question}")
        answer = ask_model(model_name, retrieved_context, question)
        time.sleep(1)

        sample_id = f"gold-{domain.replace('_', '-')}-{lang}-{i:03d}-kb-failure"

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "KNOWLEDGE_BASE_FAILURE",
            "language": lang,
            "domain": domain,
            "question": question,
            "answer": answer,
            "contexts": [retrieved_context] if retrieved_context else [],
            "model_provider": model_info["provider"],
            "model_name": model_info["name"],
            "retriever_name": RETRIEVER_NAME,
            "reference_answer": None,
            "prompt": None,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt {len(samples)} KNOWLEDGE_BASE_FAILURE candidate samples.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
