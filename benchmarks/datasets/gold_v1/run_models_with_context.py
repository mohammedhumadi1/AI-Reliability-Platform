"""
Run all questions against two real Groq models, providing the relevant
source fact as CONTEXT (simulating a real RAG pipeline).
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

MODELS = [
    {"provider": "groq", "name": "openai/gpt-oss-20b"},
    {"provider": "groq", "name": "qwen/qwen3.6-27b"},
]

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
FACTS_PATH = BASE_DIR / "source_facts.json"
OUTPUT_PATH = BASE_DIR / "model_responses.json"


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
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions_by_domain = json.load(f)
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts_by_domain = json.load(f)

    results = []
    total_calls = 0

    for domain, items in questions_by_domain.items():
        for item in items:
            fact_index = item["fact_index"]
            for lang in ("en", "ar"):
                question = item[lang]["question"]
                reference_answer = item[lang]["reference_answer"]
                context = facts_by_domain[domain][lang][fact_index]

                for model in MODELS:
                    print(f"[{domain}/{lang}/fact_{fact_index}] asking {model['name']}...")
                    try:
                        answer = ask_model(model["name"], context, question)
                        status = "ok"
                    except Exception as exc:
                        answer = ""
                        status = f"error: {exc}"

                    results.append({
                        "domain": domain,
                        "fact_index": fact_index,
                        "language": lang,
                        "context": context,
                        "question": question,
                        "reference_answer": reference_answer,
                        "model_provider": model["provider"],
                        "model_name": model["name"],
                        "model_answer": answer,
                        "status": status,
                    })
                    total_calls += 1
                    time.sleep(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone. {ok_count}/{total_calls} calls succeeded.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
