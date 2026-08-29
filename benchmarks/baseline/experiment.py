from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from benchmarks.baseline.comparison import (
    BaselineComparisonResult,
    compare_base_rag_with_platform,
)
from benchmarks.baseline.prompt import (
    PROMPT_NAME,
    PROMPT_VERSION,
    build_base_rag_prompt,
)


class GenerationResult(Protocol):
    provider: str
    model_name: str
    answer: str
    latency_seconds: float


class GenerationCallable(Protocol):
    def __call__(
        self,
        prompt: str,
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class BaselineExperimentResult:
    provider: str
    model_name: str
    latency_seconds: float
    prompt_name: str
    prompt_version: str
    prompt: str
    question: str
    contexts: tuple[str, ...]
    reference_answer: str | None
    answer: str
    comparison: BaselineComparisonResult


def run_baseline_comparison_experiment(
    *,
    project_id: str,
    question: str,
    contexts: list[str] | tuple[str, ...],
    generation_fn: GenerationCallable,
    reference_answer: str | None = None,
    comparison_fn=compare_base_rag_with_platform,
) -> BaselineExperimentResult:
    """Generate one Base-RAG answer and evaluate platform value-add."""
    prompt = build_base_rag_prompt(
        question=question,
        contexts=contexts,
    )

    generation = generation_fn(
        prompt
    )

    answer = generation.answer.strip()

    if not answer:
        raise ValueError(
            "generation returned an empty answer"
        )

    normalized_contexts = tuple(
        context.strip()
        for context in contexts
        if context.strip()
    )

    normalized_reference = (
        reference_answer.strip()
        if (
            reference_answer
            and reference_answer.strip()
        )
        else None
    )

    comparison = comparison_fn(
        project_id=project_id,
        question=question.strip(),
        answer=answer,
        contexts=list(
            normalized_contexts
        ),
        reference_answer=(
            normalized_reference
        ),
        prompt=prompt,
    )

    return BaselineExperimentResult(
        provider=generation.provider,
        model_name=generation.model_name,
        latency_seconds=(
            generation.latency_seconds
        ),
        prompt_name=PROMPT_NAME,
        prompt_version=PROMPT_VERSION,
        prompt=prompt,
        question=question.strip(),
        contexts=normalized_contexts,
        reference_answer=(
            normalized_reference
        ),
        answer=answer,
        comparison=comparison,
    )


def save_experiment_result(
    result: BaselineExperimentResult,
    output_path: str | Path,
) -> Path:
    """Persist one reproducible experiment record as UTF-8 JSON."""
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(result),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path
