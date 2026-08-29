"""Independent verification against indexed company documents."""

from __future__ import annotations

from dataclasses import dataclass
import re

from evaluation.generation.embedding_service import semantic_similarity
from evaluation.evaluators.numeric import NumericConsistencyEvaluator
from evaluation.rules.numeric_consistency import (
    extract_durations_in_days,
    extract_monetary_values,
)
from knowledge_base.vector_store import (
    collection_record_count,
    query_similar_chunks,
)


QUESTION_RELEVANCE_THRESHOLD = 0.50
ANSWER_SUPPORT_THRESHOLD = 0.60
CONTEXT_ALIGNMENT_THRESHOLD = 0.55


def _select_numeric_evidence(
    question: str,
    answer: str,
    evidence_text: str,
) -> str:
    """Return question-aligned evidence with comparable numeric content."""
    normalized = re.sub(
        r"\s+",
        " ",
        evidence_text.strip(),
    )

    if not normalized:
        return ""

    units = [
        unit.strip()
        for unit in re.split(
            r"(?<=[.!?])\s+",
            normalized,
        )
        if unit.strip()
    ]

    if not units:
        return normalized

    answer_has_duration = bool(
        extract_durations_in_days(answer)
    )
    answer_has_money = bool(
        extract_monetary_values(answer)
    )

    numeric_units = [
        unit
        for unit in units
        if (
            (
                answer_has_duration
                and extract_durations_in_days(unit)
            )
            or (
                answer_has_money
                and extract_monetary_values(unit)
            )
        )
    ]

    candidates = numeric_units or units

    return max(
        candidates,
        key=lambda unit: semantic_similarity(
            question,
            unit,
        ),
    )


@dataclass(frozen=True)
class VerificationResult:
    status: str
    evidence_found: bool
    is_supported: bool | None
    best_match_text: str
    best_match_source: str
    similarity_distance: float | None
    question_relevance_score: float | None
    answer_support_score: float | None
    context_alignment_score: float | None
    numeric_contradiction: bool
    explanation: str

    @property
    def health_score_component(
        self,
    ) -> float | None:
        """Optional 0..1 score for the overall health calculation."""
        if self.status == "NOT_AVAILABLE":
            return None

        if self.status == "NO_RELEVANT_EVIDENCE":
            return 0.0

        if self.status == "SUPPORTED":
            return 1.0

        if self.numeric_contradiction:
            return 0.0

        return self.answer_support_score


def verify_answer(
    project_id: str,
    question: str,
    answer: str,
    rag_contexts: list[str] | None = None,
    top_k: int = 3,
) -> VerificationResult:
    """
    Verify a RAG answer against independently indexed company documents.

    Retrieval is driven by the question. The generated answer is then
    compared with the best company evidence. The RAG application's own
    contexts are compared separately to the company evidence so root-cause
    analysis can distinguish retrieval failure from generation failure.

    Scores are semantic proxies, not logical entailment probabilities.
    Numeric duration contradictions receive an explicit hard check.
    """
    clean_question = question.strip()
    clean_answer = answer.strip()

    if not clean_question:
        raise ValueError(
            "question cannot be empty."
        )

    if not clean_answer:
        raise ValueError(
            "answer cannot be empty."
        )

    if collection_record_count(
        project_id
    ) == 0:
        return VerificationResult(
            status="NOT_AVAILABLE",
            evidence_found=False,
            is_supported=None,
            best_match_text="",
            best_match_source="",
            similarity_distance=None,
            question_relevance_score=None,
            answer_support_score=None,
            context_alignment_score=None,
            numeric_contradiction=False,
            explanation=(
                "No indexed company documents "
                "are available for this project."
            ),
        )

    matches = query_similar_chunks(
        project_id=project_id,
        query=clean_question,
        top_k=top_k,
    )

    if not matches:
        return VerificationResult(
            status="NO_RELEVANT_EVIDENCE",
            evidence_found=False,
            is_supported=None,
            best_match_text="",
            best_match_source="",
            similarity_distance=None,
            question_relevance_score=0.0,
            answer_support_score=None,
            context_alignment_score=None,
            numeric_contradiction=False,
            explanation=(
                "The project contains indexed documents, "
                "but no candidate evidence was retrieved."
            ),
        )

    ranked_matches = []

    for match in matches:
        evidence_text = (
            match.get("text")
            or ""
        )

        relevance = semantic_similarity(
            clean_question,
            evidence_text,
        )

        ranked_matches.append(
            (
                relevance,
                match,
            )
        )

    ranked_matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    question_relevance, best_match = (
        ranked_matches[0]
    )

    best_text = (
        best_match.get("text")
        or ""
    )

    best_source = (
        best_match.get("source")
        or ""
    )

    distance = best_match.get(
        "distance"
    )

    if (
        question_relevance
        < QUESTION_RELEVANCE_THRESHOLD
    ):
        return VerificationResult(
            status="NO_RELEVANT_EVIDENCE",
            evidence_found=False,
            is_supported=None,
            best_match_text=best_text,
            best_match_source=best_source,
            similarity_distance=distance,
            question_relevance_score=round(
                question_relevance,
                4,
            ),
            answer_support_score=None,
            context_alignment_score=None,
            numeric_contradiction=False,
            explanation=(
                "Indexed company documents exist, "
                "but the closest evidence is not "
                "sufficiently relevant to the question "
                f"(semantic relevance "
                f"{question_relevance:.2f})."
            ),
        )

    answer_support = semantic_similarity(
        clean_answer,
        best_text,
    )

    numeric_evidence = _select_numeric_evidence(
        question=clean_question,
        answer=clean_answer,
        evidence_text=best_text,
    )

    numeric_result = (
        NumericConsistencyEvaluator().evaluate(
            answer=clean_answer,
            evidence=numeric_evidence,
        )
    )

    clean_contexts = [
        context.strip()
        for context in (
            rag_contexts
            or []
        )
        if context and context.strip()
    ]

    context_alignment = (
        max(
            semantic_similarity(
                context,
                best_text,
            )
            for context in clean_contexts
        )
        if clean_contexts
        else 0.0
    )

    is_supported = (
        answer_support
        >= ANSWER_SUPPORT_THRESHOLD
        and not numeric_result.contradiction
    )

    if numeric_result.contradiction:
        status = "CONTRADICTED"
        explanation = (
            "Company evidence was found, but the generated "
            "answer contains a numeric contradiction against "
            "that evidence. "
            f"{numeric_result.explanation}"
        )

    elif is_supported:
        status = "SUPPORTED"
        explanation = (
            "The generated answer is semantically supported "
            "by independently retrieved company evidence "
            f"(answer support {answer_support:.2f})."
        )

    else:
        status = "UNSUPPORTED"
        explanation = (
            "Relevant company evidence was found, but the "
            "generated answer is not sufficiently aligned "
            f"with it (answer support "
            f"{answer_support:.2f})."
        )

    return VerificationResult(
        status=status,
        evidence_found=True,
        is_supported=is_supported,
        best_match_text=best_text,
        best_match_source=best_source,
        similarity_distance=distance,
        question_relevance_score=round(
            question_relevance,
            4,
        ),
        answer_support_score=round(
            answer_support,
            4,
        ),
        context_alignment_score=round(
            context_alignment,
            4,
        ),
        numeric_contradiction=(
            numeric_result.contradiction
        ),
        explanation=explanation,
    )
