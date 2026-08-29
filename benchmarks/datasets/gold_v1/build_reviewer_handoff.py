"""
Build the Reviewer B handoff package: anonymized candidates (no
gold_label, no label-revealing sample_id) + internal mapping (private)
+ label definitions doc.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
SAMPLES_PATH = BASE_DIR / "samples_all.json"
CANDIDATES_PATH = BASE_DIR / "reviewer_candidates.json"
MAPPING_PATH = BASE_DIR / "internal_id_mapping.json"
DEFINITIONS_PATH = BASE_DIR / "label_definitions.md"

LABEL_DEFINITIONS_MD = """\
# Gold Benchmark v1 — Label Definitions (for Reviewer B)

Please assign exactly ONE label per candidate, based only on the
question, context(s), answer, and prompt (if shown) — without seeing
any other metadata.

- **HEALTHY**: The context is relevant and sufficient, and the answer
  is correct, relevant, and supported by the context.

- **RETRIEVAL_FAILURE**: The correct information exists in the source
  knowledge, but the retrieved context(s) missed it or retrieved an
  inappropriate/irrelevant context instead.

- **GENERATION_FAILURE**: The retrieved context contains the correct
  evidence, but the answer contradicts, ignores, or hallucinates beyond
  that evidence.

- **KNOWLEDGE_BASE_FAILURE**: The information needed to answer the
  question does not exist in the available source knowledge at all.

- **PROMPT_FAILURE**: There is real evidence inside the prompt text of
  a conflict/ambiguity in instructions that caused the wrong behavior
  (only applies when a `prompt` field is shown).

If you're unsure between two labels, note it in the `notes` field along
with your reasoning.
"""


def main() -> None:
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    candidates = []
    mapping = []

    for i, s in enumerate(samples, start=1):
        neutral_id = f"candidate_{i:03d}"

        mapping.append({"neutral_id": neutral_id, "real_sample_id": s["sample_id"]})

        candidate = {
            "candidate_id": neutral_id,
            "language": s["language"],
            "domain": s["domain"],
            "question": s["question"],
            "contexts": s["contexts"],
            "answer": s["answer"],
            "prompt": s["prompt"],
            "reference_answer": s["reference_answer"],
            "model_provider": s["model_provider"],
            "model_name": s["model_name"],
            "retriever_name": s["retriever_name"],
        }
        candidates.append(candidate)

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    with open(DEFINITIONS_PATH, "w", encoding="utf-8") as f:
        f.write(LABEL_DEFINITIONS_MD)

    print(f"Built {len(candidates)} anonymized candidates.")
    print(f"-> {CANDIDATES_PATH}  (send this to Reviewer B)")
    print(f"-> {DEFINITIONS_PATH}  (send this to Reviewer B)")
    print(f"-> {MAPPING_PATH}  (KEEP PRIVATE — do not send)")


if __name__ == "__main__":
    main()
