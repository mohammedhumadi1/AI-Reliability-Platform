import pytest

from benchmarks.validation_coverage import (
    CoverageRequirements,
    calculate_validation_coverage,
    validate_benchmark_coverage,
)
from benchmarks.validation_schema import (
    FailureLabel,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


def _benchmark_samples() -> list[ValidationSample]:
    samples = []

    for label_index, label in enumerate(
        FailureLabel
    ):
        for index in range(2):
            is_arabic = (
                (label_index + index) % 2 == 0
            )

            samples.append(
                ValidationSample(
                    sample_id=(
                        f"{label.value}-{index}"
                    ),
                    split=(
                        ValidationSplit.DEVELOPMENT
                        if index == 0
                        else ValidationSplit.HELD_OUT
                    ),
                    gold_label=label,
                    language=(
                        SupportedLanguage.ARABIC
                        if is_arabic
                        else SupportedLanguage.ENGLISH
                    ),
                    domain=(
                        "finance"
                        if index == 0
                        else "support"
                    ),
                    question="Question?",
                    answer="Answer.",
                    contexts=("Context.",),
                    model_provider=(
                        "provider-a"
                        if index == 0
                        else "provider-b"
                    ),
                    model_name=(
                        "model-a"
                        if index == 0
                        else "model-b"
                    ),
                    retriever_name=(
                        "retriever-a"
                        if index == 0
                        else "retriever-b"
                    ),
                    prompt=(
                        "Use only the provided context."
                        if (
                            label
                            == FailureLabel.PROMPT_FAILURE
                        )
                        else None
                    ),
                )
            )

    return samples


def test_calculate_validation_coverage() -> None:
    coverage = calculate_validation_coverage(
        _benchmark_samples()
    )

    assert coverage.sample_count == 10

    for label in FailureLabel:
        assert (
            coverage.by_label[label.value]
            == 2
        )

    assert coverage.by_language["ar"] == 5
    assert coverage.by_language["en"] == 5
    assert len(coverage.by_domain) == 2
    assert len(coverage.by_model) == 2
    assert len(coverage.by_retriever) == 2
    assert coverage.by_split["development"] == 5
    assert coverage.by_split["held_out"] == 5


def test_valid_benchmark_meets_default_policy() -> None:
    coverage = validate_benchmark_coverage(
        _benchmark_samples()
    )

    assert coverage.sample_count == 10


def test_coverage_rejects_missing_language() -> None:
    samples = [
        ValidationSample(
            sample_id=sample.sample_id,
            split=sample.split,
            gold_label=sample.gold_label,
            language=SupportedLanguage.ENGLISH,
            domain=sample.domain,
            question=sample.question,
            answer=sample.answer,
            contexts=sample.contexts,
            model_provider=sample.model_provider,
            model_name=sample.model_name,
            retriever_name=sample.retriever_name,
            prompt=sample.prompt,
        )
        for sample in _benchmark_samples()
    ]

    with pytest.raises(
        ValueError,
        match="both Arabic and English",
    ):
        validate_benchmark_coverage(
            samples
        )


def test_coverage_rejects_unspecified_model() -> None:
    samples = _benchmark_samples()

    first = samples[0]

    samples[0] = ValidationSample(
        sample_id=first.sample_id,
        split=first.split,
        gold_label=first.gold_label,
        language=first.language,
        domain=first.domain,
        question=first.question,
        answer=first.answer,
        contexts=first.contexts,
        model_provider="unspecified",
        model_name=first.model_name,
        retriever_name=first.retriever_name,
        prompt=first.prompt,
    )

    with pytest.raises(
        ValueError,
        match="unspecified model metadata",
    ):
        validate_benchmark_coverage(
            samples
        )


def test_coverage_rejects_unspecified_retriever() -> None:
    samples = _benchmark_samples()

    first = samples[0]

    samples[0] = ValidationSample(
        sample_id=first.sample_id,
        split=first.split,
        gold_label=first.gold_label,
        language=first.language,
        domain=first.domain,
        question=first.question,
        answer=first.answer,
        contexts=first.contexts,
        model_provider=first.model_provider,
        model_name=first.model_name,
        retriever_name="unspecified",
        prompt=first.prompt,
    )

    with pytest.raises(
        ValueError,
        match="unspecified retriever metadata",
    ):
        validate_benchmark_coverage(
            samples
        )


def test_coverage_policy_is_configurable() -> None:
    with pytest.raises(
        ValueError,
        match="LLM diversity",
    ):
        validate_benchmark_coverage(
            _benchmark_samples(),
            requirements=CoverageRequirements(
                minimum_samples_per_label=2,
                minimum_languages=2,
                minimum_domains=2,
                minimum_models=3,
                minimum_retrievers=2,
            ),
        )
