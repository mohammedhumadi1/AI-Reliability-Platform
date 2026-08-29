"""
Build 10 PROMPT_FAILURE candidate samples using a genuinely
self-contradictory prompt, run for real against real models.
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
RESPONSES_PATH = BASE_DIR / "model_responses.json"
OUTPUT_PATH = BASE_DIR / "samples_prompt_failure.json"

PROMPT_FAILURE_SELECTION = [
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

CONFLICTING_PROMPT_EN = (
    "You are a company assistant.\n"
    "Rule 1: Answer strictly and only using the information in the "
    "Context below. Do not use any outside knowledge.\n"
    "Rule 2: You are also encouraged to supplement your answer with "
    "general industry-standard practices, even if they are not stated "
    "in the Context, in order to be more helpful.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer directly in one sentence."
)

CONFLICTING_PROMPT_AR = (
    "أنت مساعد افتراضي لشركة.\n"
    "القاعدة 1: أجب بدقة باستخدام المعلومات الموجودة في السياق أدناه فقط. "
    "لا تستخدم أي معرفة خارجية.\n"
    "القاعدة 2: يُشجَّع أيضًا أن تضيف إلى إجابتك ممارسات شائعة في الصناعة "
    "بشكل عام، حتى لو لم تكن مذكورة في السياق، لتكون أكثر فائدة.\n\n"
    "السياق: {context}\n\n"
    "السؤال: {question}\n\n"
    "أجب مباشرة في جملة واحدة."
)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(model_name: str, prompt_text: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=1200,
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    return strip_thinking(raw)


def main() -> None:
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)

    lookup = {(r["domain"], r["fact_index"], r["language"]): r for r in responses}

    models_cycle = ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
    model_providers = {"openai/gpt-oss-20b": "groq", "qwen/qwen3.6-27b": "groq"}

    samples = []

    for i, (domain, fact_index, lang) in enumerate(PROMPT_FAILURE_SELECTION):
        q_info = lookup[(domain, fact_index, lang)]
        context = q_info["context"]
        question = q_info["question"]

        template = CONFLICTING_PROMPT_AR if lang == "ar" else CONFLICTING_PROMPT_EN
        full_prompt = template.format(context=context, question=question)

        model_name = models_cycle[i % 2]
        print(f"[{domain}/{lang}/fact_{fact_index}] asking {model_name} (conflicting prompt)...")
        answer = ask_model(model_name, full_prompt)
        time.sleep(1)

        sample_id = f"gold-{domain.replace('_', '-')}-{lang}-{fact_index:03d}-prompt-failure"

        sample = {
            "sample_id": sample_id,
            "split": "development",
            "gold_label": "PROMPT_FAILURE",
            "language": lang,
            "domain": domain,
            "question": question,
            "answer": answer,
            "contexts": [context],
            "model_provider": model_providers[model_name],
            "model_name": model_name,
            "retriever_name": "chroma_minilm_topk1",
            "reference_answer": q_info["reference_answer"],
            "prompt": full_prompt,
            "reviewers": [],
            "adjudication": None,
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt {len(samples)} PROMPT_FAILURE candidate samples.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
