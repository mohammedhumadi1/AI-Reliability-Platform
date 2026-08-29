"""Run deterministic root-cause rules in evidence-aware priority order."""

from __future__ import annotations

from typing import Optional

from root_cause.rules.retrieval_rules import (
    check_generation_hallucination,
    check_knowledge_gap,
    check_prompt_failure,
    check_retrieval_failure,
    check_verified_knowledge_result,
)


def run_rules_pipeline(
    metrics: dict,
) -> Optional[dict]:
    """
    Prefer direct company-KB evidence when available, then fall back
    to proxy metrics from the regular evaluation pipeline.
    """
    verification_status = metrics.get(
        "verification_status"
    )

    if verification_status is not None:
        verified_diagnosis = (
            check_verified_knowledge_result(
                verification_status=(
                    verification_status
                ),
                context_alignment_score=(
                    metrics.get(
                        "context_alignment_score"
                    )
                ),
                explanation=(
                    metrics.get(
                        "verification_explanation",
                        "",
                    )
                ),
            )
        )

        if verified_diagnosis is not None:
            return verified_diagnosis

    context_precision = metrics.get(
        "context_precision"
    )

    faithfulness = metrics.get(
        "faithfulness"
    )

    if context_precision is None:
        return None

    retrieval_diagnosis = (
        check_retrieval_failure(
            context_precision
        )
    )

    if retrieval_diagnosis is not None:
        return retrieval_diagnosis

    if faithfulness is not None:
        hallucination_diagnosis = (
            check_generation_hallucination(
                context_precision=(
                    context_precision
                ),
                faithfulness=faithfulness,
            )
        )

        if hallucination_diagnosis is not None:
            return hallucination_diagnosis

    context_recall = metrics.get(
        "context_recall"
    )

    if context_recall is not None:
        knowledge_gap_diagnosis = (
            check_knowledge_gap(
                context_recall=context_recall,
                context_precision=(
                    context_precision
                ),
            )
        )

        if knowledge_gap_diagnosis is not None:
            return knowledge_gap_diagnosis

    answer_relevancy = metrics.get(
        "answer_relevancy"
    )

    if answer_relevancy is not None:
        prompt_failure_diagnosis = (
            check_prompt_failure(
                context_precision=(
                    context_precision
                ),
                answer_relevancy=(
                    answer_relevancy
                ),
                prompt_evidence=(
                    metrics.get(
                        "prompt_evidence"
                    )
                ),
            )
        )

        if prompt_failure_diagnosis is not None:
            return prompt_failure_diagnosis

    return None
