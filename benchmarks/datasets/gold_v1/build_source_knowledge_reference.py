"""
Build a source knowledge reference document for Reviewer B.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
FACTS_PATH = BASE_DIR / "source_facts.json"
OUTPUT_PATH = BASE_DIR / "source_knowledge_reference.md"

DOMAIN_LABELS = {
    "it_support": "IT Support",
    "travel_expense": "Travel / Expense",
    "customer_returns": "Customer Returns",
}


def main() -> None:
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts_by_domain = json.load(f)

    lines = [
        "# Source Knowledge Reference (for Reviewer B)",
        "",
        "This document lists EVERY fact available in the source knowledge",
        "base used to build the candidates. Use it to check whether the",
        "information needed to answer a candidate's question actually",
        "exists here or not — this distinguishes:",
        "",
        "- **RETRIEVAL_FAILURE**: the needed fact IS listed below, but the",
        "  candidate's `contexts` field doesn't contain it.",
        "- **KNOWLEDGE_BASE_FAILURE**: the needed fact is NOT listed below",
        "  at all, in either language.",
        "",
        "---",
        "",
    ]

    for domain, langs in facts_by_domain.items():
        label = DOMAIN_LABELS.get(domain, domain)
        lines.append(f"## {label}")
        lines.append("")
        en_facts = langs.get("en", [])
        ar_facts = langs.get("ar", [])
        for i, (en, ar) in enumerate(zip(en_facts, ar_facts)):
            lines.append(f"{i + 1}. {en}")
            lines.append(f"   {ar}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "If a candidate's question is about something NOT covered above "
        "(in either language), the required information does not exist "
        "in the source knowledge at all."
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Built source knowledge reference document.")
    print(f"-> {OUTPUT_PATH}  (send this to Reviewer B alongside the candidates)")


if __name__ == "__main__":
    main()
