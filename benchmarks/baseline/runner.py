from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from benchmarks.baseline.prompt import (
    PROMPT_NAME,
    PROMPT_VERSION,
    build_base_rag_prompt,
)


LLMCallable = Callable[[str], str]


@dataclass(frozen=True)
class BaseRAGResult:
    question: str
    contexts: tuple[str, ...]
    prompt_name: str
    prompt_version: str
    prompt: str
    answer: str


def run_base_rag(
    question: str,
    contexts: list[str] | tuple[str, ...],
    llm: LLMCallable,
) -> BaseRAGResult:
    """Run the Base-RAG benchmark with the versioned prompt."""
    prompt = build_base_rag_prompt(
        question=question,
        contexts=contexts,
    )

    answer = llm(prompt).strip()

    if not answer:
        raise ValueError("llm returned an empty answer")

    return BaseRAGResult(
        question=question.strip(),
        contexts=tuple(
            context.strip()
            for context in contexts
            if context.strip()
        ),
        prompt_name=PROMPT_NAME,
        prompt_version=PROMPT_VERSION,
        prompt=prompt,
        answer=answer,
    )
