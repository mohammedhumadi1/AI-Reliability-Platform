import json
from dataclasses import dataclass

import pytest

from benchmarks.baseline.comparison import (
    BaselineComparisonResult,
    CoreMetricSnapshot,
    PlatformValueAdd,
)
from benchmarks.baseline.experiment import (
    run_baseline_comparison_experiment,
    save_experiment_result,
)


@dataclass(frozen=True)
class FakeGeneration:
    provider: str = "groq"
    model_name: str = "test-model"
    answer: str = " Grounded answer. "
    latency_seconds: float = 1.25


def make_comparison():
    metrics = CoreMetricSnapshot(
        correctness=1.0,
        faithfulness=0.9,
        context_precision=0.8,
        context_recall=0.7,
        answer_relevancy=0.9,
        hallucination_risk=0.1,
    )

    value_add = PlatformValueAdd(
        verification_status="NOT_AVAILABLE",
        kb_evidence_found=False,
        knowledge_base_support=None,
        diagnosis_category=None,
        diagnosis_subcategory=None,
        diagnosis_severity=None,
        diagnosis_confidence=None,
        health_score=86,
        health_status="GOOD",
        recommendation_count=0,
        recommendation_actions=(),
    )

    return BaselineComparisonResult(
        base_rag_metrics=metrics,
        full_platform_metrics=metrics,
        core_metric_deltas={
            key: 0.0
            for key in metrics.as_dict()
        },
        platform_value_add=value_add,
    )


def test_experiment_records_generation_and_prompt():
    captured = {}

    def fake_generation(prompt):
        captured["prompt"] = prompt
        return FakeGeneration()

    def fake_comparison(**kwargs):
        captured["comparison"] = kwargs
        return make_comparison()

    result = (
        run_baseline_comparison_experiment(
            project_id="project-1",
            question=" What is RAG? ",
            contexts=[
                " First context. ",
                "",
            ],
            reference_answer=(
                " Reference. "
            ),
            generation_fn=fake_generation,
            comparison_fn=fake_comparison,
        )
    )

    assert result.provider == "groq"
    assert result.model_name == "test-model"
    assert result.latency_seconds == 1.25
    assert result.prompt_name == "base-rag"
    assert result.prompt_version == "base-rag-v1"
    assert result.answer == "Grounded answer."
    assert result.contexts == (
        "First context.",
    )

    assert (
        captured["comparison"]["prompt"]
        == captured["prompt"]
    )
    assert (
        captured["comparison"]["answer"]
        == "Grounded answer."
    )


def test_experiment_rejects_empty_generation():
    def fake_generation(prompt):
        return FakeGeneration(
            answer="   "
        )

    with pytest.raises(
        ValueError,
        match=(
            "generation returned an empty answer"
        ),
    ):
        run_baseline_comparison_experiment(
            project_id="project-1",
            question="What is RAG?",
            contexts=["Context."],
            generation_fn=fake_generation,
            comparison_fn=(
                lambda **kwargs: (
                    make_comparison()
                )
            ),
        )


def test_experiment_result_can_be_saved(
    tmp_path,
):
    result = (
        run_baseline_comparison_experiment(
            project_id="project-1",
            question="What is RAG?",
            contexts=["Context."],
            generation_fn=(
                lambda prompt: (
                    FakeGeneration()
                )
            ),
            comparison_fn=(
                lambda **kwargs: (
                    make_comparison()
                )
            ),
        )
    )

    output_path = (
        tmp_path
        / "experiment.json"
    )

    saved = save_experiment_result(
        result,
        output_path,
    )

    payload = json.loads(
        saved.read_text(
            encoding="utf-8"
        )
    )

    assert payload["provider"] == "groq"
    assert (
        payload["prompt_version"]
        == "base-rag-v1"
    )
    assert (
        payload["comparison"]
        ["platform_value_add"]
        ["health_status"]
        == "GOOD"
    )
