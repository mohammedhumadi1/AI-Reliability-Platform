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
    Prefer direct evidence over proxy-only metrics.

    A verified absence of relevant company knowledge remains
    the strongest KB-failure signal. When company knowledge
    exists, direct prompt-conflict evidence may identify a
    prompt failure before downstream answer symptoms are
    attributed to generation.
    """
    verification_status = metrics.get(
        "verification_status"
    )

    context_precision = metrics.get(
        "context_precision"
    )

    answer_relevancy = metrics.get(
        "answer_relevancy"
    )

    # A verified absence of relevant company evidence cannot
    # be repaired by prompt or generation changes.
    if verification_status == "NO_RELEVANT_EVIDENCE":
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

    # Direct conflicting prompt evidence is stronger than
    # downstream semantic symptoms when retrieval is adequate.
    if (
        context_precision is not None
        and answer_relevancy is not None
    ):
        prompt_diagnosis = (
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

        if prompt_diagnosis is not None:
            return prompt_diagnosis

    # Use remaining direct KB verification outcomes before
    # proxy-only retrieval/generation rules.
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

    if context_precision is None:
        return None

    retrieval_diagnosis = (
        check_retrieval_failure(
            context_precision
        )
    )

    if retrieval_diagnosis is not None:
        return retrieval_diagnosis

    faithfulness = metrics.get(
        "faithfulness"
    )

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

    # When independent KB verification explicitly says the
    # answer is supported, a low proxy context-recall score
    # must not override that direct evidence as a KB failure.
    if verification_status != "SUPPORTED":
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

    return None
