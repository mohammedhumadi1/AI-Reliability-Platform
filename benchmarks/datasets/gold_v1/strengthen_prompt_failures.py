"""
Re-run the 6 weak PROMPT_FAILURE samples with a stronger conflicting
instruction to get genuine evidence of prompt-caused deviation.
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
client = Groq(api_key=api_key)

BASE_DIR = Path(__file__).parent
RESPONSES_PATH = BASE_DIR / "model_responses.json"
SAMPLES_PATH = BASE_DIR / "samples_prompt_failure.json"

WEAK_SAMPLE_IDS = {
    "gold-it-support-ar-003-prompt-failure",
    "gold-customer-returns-ar-003-prompt-failure",
}

STRONGER_PROMPT_EN = (
    "You are a company assistant.\n"
    "Rule 1: Answer strictly and only using the information in the "
    "Context below. Do not use any outside knowledge.\n"
    "Rule 2: You must always add at least one extra detail, number, or "
    "typical industry benchmark that is NOT stated in the Context, so "
    "the answer feels more complete and helpful.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer directly in one sentence."
)

STRONGER_PROMPT_AR = (
    "أنت مساعد افتراضي لشركة.\n"
    "القاعدة 1: أجب بدقة باستخدام المعلومات الموجودة في السياق أدناه "
    "فقط. لا تستخدم أي معرفة خارجية.\n"
    "القاعدة 2: يجب عليك دائمًا إضافة تفصيل أو رقم إضافي أو معيار شائع "
    "في الصناعة غير مذكور في السياق، حتى تبدو الإجابة أكثر اكتمالًا "
    "وفائدة.\n\n"
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
        max_tokens=3000,
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    return strip_thinking(raw)


def main() -> None:
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        responses = json.load(f)
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    lookup = {(r["domain"], r["fact_index"], r["language"]): r for r in responses}

    updated = 0
    for sample in samples:
        if sample["sample_id"] not in WEAK_SAMPLE_IDS:
            continue

        domain = sample["domain"]
        lang = sample["language"]
        fact_index = int(sample["sample_id"].split("-")[-3])

        q_info = lookup[(domain, fact_index, lang)]
        context = q_info["context"]
        question = q_info["question"]

        template = STRONGER_PROMPT_AR if lang == "ar" else STRONGER_PROMPT_EN
        full_prompt = template.format(context=context, question=question)

        model_name = sample["model_name"]
        print(f"Re-running {sample['sample_id']} with stronger conflict...")
        answer = ask_model(model_name, full_prompt)
        time.sleep(1)

        sample["answer"] = answer
        sample["prompt"] = full_prompt
        updated += 1

    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {updated} samples with stronger prompt conflict evidence.")
    print(f"Saved to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
