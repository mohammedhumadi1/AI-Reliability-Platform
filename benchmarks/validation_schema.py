from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureLabel(StrEnum):
    HEALTHY = "HEALTHY"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    KNOWLEDGE_BASE_FAILURE = "KNOWLEDGE_BASE_FAILURE"
    PROMPT_FAILURE = "PROMPT_FAILURE"


class ValidationSplit(StrEnum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class SupportedLanguage(StrEnum):
    ARABIC = "ar"
    ENGLISH = "en"


@dataclass(frozen=True)
class ReviewerAnnotation:
    reviewer_id: str
    label: FailureLabel
    notes: str | None = None

    def __post_init__(self) -> None:
        reviewer_id = self.reviewer_id.strip()

        if not reviewer_id:
            raise ValueError(
                "reviewer_id cannot be empty."
            )

        object.__setattr__(
            self,
            "reviewer_id",
            reviewer_id,
        )

        if self.notes is not None:
            notes = self.notes.strip()

            object.__setattr__(
                self,
                "notes",
                notes or None,
            )


@dataclass(frozen=True)
class Adjudication:
    adjudicator_id: str
    label: FailureLabel
    notes: str | None = None

    def __post_init__(self) -> None:
        adjudicator_id = self.adjudicator_id.strip()

        if not adjudicator_id:
            raise ValueError(
                "adjudicator_id cannot be empty."
            )

        object.__setattr__(
            self,
            "adjudicator_id",
            adjudicator_id,
        )

        if self.notes is not None:
            notes = self.notes.strip()

            object.__setattr__(
                self,
                "notes",
                notes or None,
            )


@dataclass(frozen=True)
class ValidationSample:
    sample_id: str
    split: ValidationSplit
    gold_label: FailureLabel
    language: SupportedLanguage
    domain: str
    question: str
    answer: str
    contexts: tuple[str, ...]
    model_provider: str = "unspecified"
    model_name: str = "unspecified"
    retriever_name: str = "unspecified"
    reference_answer: str | None = None
    prompt: str | None = None
    reviewers: tuple[
        ReviewerAnnotation,
        ...
    ] = ()
    adjudication: Adjudication | None = None
    source_fact_id: str | None = None

    def __post_init__(self) -> None:
        text_fields = {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "question": self.question,
            "answer": self.answer,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "retriever_name": self.retriever_name,
        }

        for name, value in text_fields.items():
            cleaned = value.strip()

            if not cleaned:
                raise ValueError(
                    f"{name} cannot be empty."
                )

            object.__setattr__(
                self,
                name,
                cleaned,
            )

        clean_contexts = tuple(
            context.strip()
            for context in self.contexts
            if context and context.strip()
        )

        object.__setattr__(
            self,
            "contexts",
            clean_contexts,
        )

        if self.reference_answer is not None:
            reference = (
                self.reference_answer.strip()
            )

            object.__setattr__(
                self,
                "reference_answer",
                reference or None,
            )

        if self.prompt is not None:
            prompt = self.prompt.strip()

            object.__setattr__(
                self,
                "prompt",
                prompt or None,
            )

        reviewer_ids = [
            annotation.reviewer_id
            for annotation in self.reviewers
        ]

        if len(reviewer_ids) != len(
            set(reviewer_ids)
        ):
            raise ValueError(
                "Reviewer IDs must be unique "
                "within a sample."
            )
