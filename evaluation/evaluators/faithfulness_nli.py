import re
from dataclasses import dataclass

from evaluation.generation.nli_service import (
    evaluate_nli_batch,
)


@dataclass(frozen=True)
class ClaimSupportResult:
    claim: str
    evidence: str
    entailment: float
    neutral: float
    contradiction: float
    label: str


@dataclass(frozen=True)
class NLIFaithfulnessResult:
    score: float
    claims: tuple[ClaimSupportResult, ...]
    explanation: str


class NLIFaithfulnessEvaluator:
    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?؟])\s+|\n+",
                text.strip(),
            )
            if sentence.strip()
        ]

    def evaluate(
        self,
        answer: str,
        contexts: list[str],
    ) -> NLIFaithfulnessResult:
        answer_sentences = (
            self._split_sentences(
                answer
            )
        )

        context_sentences = []

        for context in contexts:
            if not context:
                continue

            context_sentences.extend(
                self._split_sentences(
                    context
                )
            )

        if not answer_sentences:
            return NLIFaithfulnessResult(
                score=0.0,
                claims=(),
                explanation=(
                    "No usable claims were found "
                    "in the answer."
                ),
            )

        if not context_sentences:
            return NLIFaithfulnessResult(
                score=0.0,
                claims=(),
                explanation=(
                    "No usable evidence was found "
                    "in the retrieved context."
                ),
            )

        claim_results = []

        for claim in answer_sentences:
            pairs = [
                (
                    evidence,
                    claim,
                )
                for evidence
                in context_sentences
            ]

            nli_results = evaluate_nli_batch(
                pairs
            )

            best_index = max(
                range(len(nli_results)),
                key=lambda index: (
                    nli_results[
                        index
                    ].entailment
                ),
            )

            best = nli_results[
                best_index
            ]

            claim_results.append(
                ClaimSupportResult(
                    claim=claim,
                    evidence=(
                        context_sentences[
                            best_index
                        ]
                    ),
                    entailment=(
                        best.entailment
                    ),
                    neutral=best.neutral,
                    contradiction=(
                        best.contradiction
                    ),
                    label=best.label,
                )
            )

        score = sum(
            result.entailment
            for result in claim_results
        ) / len(claim_results)

        return NLIFaithfulnessResult(
            score=round(score, 4),
            claims=tuple(
                claim_results
            ),
            explanation=(
                "Faithfulness was evaluated "
                "claim-by-claim using NLI "
                "entailment against retrieved "
                "evidence."
            ),
        )