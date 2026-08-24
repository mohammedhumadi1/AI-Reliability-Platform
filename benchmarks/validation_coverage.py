from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from benchmarks.validation_schema import (
    FailureLabel,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class CoverageRequirements:
    minimum_samples_per_label: int = 2
    minimum_languages: int = 2
    minimum_domains: int = 2
    minimum_models: int = 2
    minimum_retrievers: int = 2


@dataclass(frozen=True)
class ValidationCoverage:
    sample_count: int
    by_label: dict[str, int]
    by_language: dict[str, int]
    by_domain: dict[str, int]
    by_model: dict[str, int]
    by_retriever: dict[str, int]
    by_split: dict[str, int]


def _model_key(
    sample: ValidationSample,
) -> str:
    return (
        f"{sample.model_provider}/"
        f"{sample.model_name}"
    )


def calculate_validation_coverage(
    samples: list[ValidationSample],
) -> ValidationCoverage:
    if not samples:
        raise ValueError(
            "At least one validation sample is required."
        )

    return ValidationCoverage(
        sample_count=len(samples),
        by_label=dict(
            Counter(
                sample.gold_label.value
                for sample in samples
            )
        ),
        by_language=dict(
            Counter(
                sample.language.value
                for sample in samples
            )
        ),
        by_domain=dict(
            Counter(
                sample.domain
                for sample in samples
            )
        ),
        by_model=dict(
            Counter(
                _model_key(sample)
                for sample in samples
            )
        ),
        by_retriever=dict(
            Counter(
                sample.retriever_name
                for sample in samples
            )
        ),
        by_split=dict(
            Counter(
                sample.split.value
                for sample in samples
            )
        ),
    )


def validate_benchmark_coverage(
    samples: list[ValidationSample],
    requirements: CoverageRequirements = (
        CoverageRequirements()
    ),
) -> ValidationCoverage:
    coverage = calculate_validation_coverage(
        samples
    )

    if requirements.minimum_samples_per_label < 1:
        raise ValueError(
            "minimum_samples_per_label must be at least 1."
        )

    if requirements.minimum_languages < 1:
        raise ValueError(
            "minimum_languages must be at least 1."
        )

    if requirements.minimum_domains < 1:
        raise ValueError(
            "minimum_domains must be at least 1."
        )

    if requirements.minimum_models < 1:
        raise ValueError(
            "minimum_models must be at least 1."
        )

    if requirements.minimum_retrievers < 1:
        raise ValueError(
            "minimum_retrievers must be at least 1."
        )

    missing_labels = [
        label.value
        for label in FailureLabel
        if coverage.by_label.get(
            label.value,
            0,
        )
        < requirements.minimum_samples_per_label
    ]

    if missing_labels:
        raise ValueError(
            "Insufficient samples for labels: "
            + ", ".join(missing_labels)
        )

    required_languages = {
        SupportedLanguage.ARABIC.value,
        SupportedLanguage.ENGLISH.value,
    }

    present_languages = set(
        coverage.by_language
    )

    if not required_languages.issubset(
        present_languages
    ):
        raise ValueError(
            "Benchmark must contain both Arabic "
            "and English samples."
        )

    if (
        len(present_languages)
        < requirements.minimum_languages
    ):
        raise ValueError(
            "Benchmark does not meet the minimum "
            "language diversity requirement."
        )

    if (
        len(coverage.by_domain)
        < requirements.minimum_domains
    ):
        raise ValueError(
            "Benchmark does not meet the minimum "
            "domain diversity requirement."
        )

    unspecified_models = [
        sample.sample_id
        for sample in samples
        if (
            sample.model_provider.lower()
            == UNSPECIFIED
            or sample.model_name.lower()
            == UNSPECIFIED
        )
    ]

    if unspecified_models:
        raise ValueError(
            "Benchmark contains samples with "
            "unspecified model metadata: "
            + ", ".join(
                unspecified_models
            )
        )

    unspecified_retrievers = [
        sample.sample_id
        for sample in samples
        if (
            sample.retriever_name.lower()
            == UNSPECIFIED
        )
    ]

    if unspecified_retrievers:
        raise ValueError(
            "Benchmark contains samples with "
            "unspecified retriever metadata: "
            + ", ".join(
                unspecified_retrievers
            )
        )

    if (
        len(coverage.by_model)
        < requirements.minimum_models
    ):
        raise ValueError(
            "Benchmark does not meet the minimum "
            "LLM diversity requirement."
        )

    if (
        len(coverage.by_retriever)
        < requirements.minimum_retrievers
    ):
        raise ValueError(
            "Benchmark does not meet the minimum "
            "retriever diversity requirement."
        )

    allowed_splits = {
        ValidationSplit.DEVELOPMENT.value,
        ValidationSplit.HELD_OUT.value,
    }

    if not set(
        coverage.by_split
    ).issubset(allowed_splits):
        raise ValueError(
            "Benchmark contains an unknown split."
        )

    return coverage
