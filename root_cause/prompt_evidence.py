from __future__ import annotations

import re
from typing import Optional


ISSUE_CODE = "CONFLICTING_GROUNDING_INSTRUCTIONS"

# Deterministic rule confidence, not an empirical probability.
CONFIDENCE = 0.95


def _normalize_prompt(prompt: str) -> str:
    text = (
        prompt.lower()
        .replace("\u2019", "'")
    )

    text = re.sub(
        r"\bdon't\b",
        "do not",
        text,
    )

    return " ".join(
        text.split()
    )


def _matches_any(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def _remove_negated_conflicts(
    text: str,
) -> str:
    patterns = (
        (
            r"\bdo\s+not"
            r"(?:\s+\w+){0,4}\s+"
            r"ignore\s+(?:the\s+)?provided\s+context\b"
        ),
        (
            r"\bnever"
            r"(?:\s+\w+){0,4}\s+"
            r"ignore\s+(?:the\s+)?provided\s+context\b"
        ),
        (
            r"\bdo\s+not"
            r"(?:\s+\w+){0,4}\s+"
            r"use\s+your\s+own\s+knowledge\b"
        ),
        (
            r"\bnever"
            r"(?:\s+\w+){0,4}\s+"
            r"use\s+your\s+own\s+knowledge\b"
        ),
        (
            r"\bdo\s+not"
            r"(?:\s+\w+){0,4}\s+"
            r"use\s+external\s+knowledge\b"
        ),
        (
            r"\bnever"
            r"(?:\s+\w+){0,4}\s+"
            r"use\s+external\s+knowledge\b"
        ),
    )

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
        )

    return cleaned


def analyze_prompt_evidence(
    prompt: str | None,
) -> Optional[dict]:
    """
    Detect high-confidence contradictory grounding
    instructions in English or Arabic prompts.

    Absence of grounding instructions is not itself
    considered a prompt failure.
    """
    if prompt is None:
        return None

    text = _normalize_prompt(prompt)

    if not text:
        return None

    grounding_patterns = (
        (
            r"\banswer\s+only\s+from\s+"
            r"(?:the\s+)?provided\s+context\b"
        ),
        (
            r"\banswer\s+the\s+user(?:'s)?\s+question\s+"
            r"using\s+only\s+(?:the\s+)?provided\s+context\b"
        ),
        (
            r"\busing\s+only\s+(?:the\s+)?provided\s+context\b"
        ),
        (
            r"\bdo\s+not\s+use\s+external\s+knowledge\b"
        ),
        (
            r"\u0623\u062c\u0628\s+\u0641\u0642\u0637\s+"
            r"\u0645\u0646\s+\u0627\u0644\u0633\u064a\u0627\u0642"
            r"(?:\s+\u0627\u0644\u0645\u0642\u062f\u0645)?"
        ),
        (
            r"\u0627\u0639\u062a\u0645\u062f\s+\u0641\u0642\u0637\s+"
            r"\u0639\u0644\u0649\s+\u0627\u0644\u0633\u064a\u0627\u0642"
            r"(?:\s+\u0627\u0644\u0645\u0642\u062f\u0645)?"
        ),
        (
            r"\u0644\u0627\s+\u062a\u0633\u062a\u062e\u062f\u0645\s+"
            r"(?:\u0623\u064a\s+)?"
            r"\u0645\u0639\u0631\u0641\u0629\s+"
            r"\u062e\u0627\u0631\u062c\u064a\u0629"
        ),
    )

    conflict_text = _remove_negated_conflicts(
        text
    )

    conflicting_patterns = (
        (
            r"\bignore\s+(?:the\s+)?provided\s+context\b"
        ),
        (
            r"\buse\s+your\s+own\s+knowledge\b"
        ),
        (
            r"\buse\s+external\s+knowledge\b"
        ),
        (
            r"\b(?:\u0648)?"
            r"\u062a\u062c\u0627\u0647\u0644\s+"
            r"\u0627\u0644\u0633\u064a\u0627\u0642"
            r"(?:\s+\u0627\u0644\u0645\u0642\u062f\u0645)?"
        ),
        (
            r"\u0627\u0633\u062a\u062e\u062f\u0645\s+"
            r"\u0645\u0639\u0631\u0641\u062a\u0643\s+"
            r"\u0627\u0644\u062e\u0627\u0635\u0629"
        ),
        (
            r"\u0627\u0633\u062a\u062e\u062f\u0645\s+"
            r"\u0645\u0639\u0631\u0641\u0629\s+"
            r"\u062e\u0627\u0631\u062c\u064a\u0629"
        ),
    )

    has_grounding_instruction = _matches_any(
        text,
        grounding_patterns,
    )

    has_conflicting_instruction = _matches_any(
        conflict_text,
        conflicting_patterns,
    )

    if not (
        has_grounding_instruction
        and has_conflicting_instruction
    ):
        return None

    return {
        "issue_code": ISSUE_CODE,
        "confidence": CONFIDENCE,
        "explanation": (
            "The prompt contains conflicting grounding "
            "instructions: it both restricts the answer "
            "to supplied context and permits or instructs "
            "the model to ignore that grounding."
        ),
    }
