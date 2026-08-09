import re

from evaluation.evaluators.models import (
    ScoreResult,
)
from evaluation.generation.embedding_service import (
    semantic_similarity,
)


class FaithfulnessEvaluator:
    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?؟])\s+",
                text.strip(),
            )
            if sentence.strip()
        ]

    def evaluate(
        self,
        answer: str,
        contexts: list[str] | None = None,
        combined_context: str | None = None,
    ) -> ScoreResult:
        clean_contexts = [
            context.strip()
            for context in (contexts or [])
            if context and context.strip()
        ]

        # Backward compatibility.
        if not clean_contexts and combined_context:
            clean_combined_context = combined_context.strip()

            if clean_combined_context:
                clean_contexts = [
                    clean_combined_context
                ]

        if not clean_contexts:
            return ScoreResult(
                score=0.0,
                explanation=(
                    "Faithfulness could not be "
                    "evaluated because no retrieved "
                    "context was provided."
                ),
            )

        answer_sentences = self._split_sentences(
            answer
        )

        context_sentences = []

        for context in clean_contexts:
            context_sentences.extend(
                self._split_sentences(
                    context
                )
            )

        if not answer_sentences:
            return ScoreResult(
                score=0.0,
                explanation=(
                    "Faithfulness could not be "
                    "evaluated because the answer "
                    "contained no usable sentences."
                ),
            )

        if not context_sentences:
            return ScoreResult(
                score=0.0,
                explanation=(
                    "Faithfulness could not be "
                    "evaluated because the retrieved "
                    "context contained no usable "
                    "sentences."
                ),
            )

        sentence_scores = []

        for answer_sentence in answer_sentences:
            evidence_scores = [
                semantic_similarity(
                    answer_sentence,
                    context_sentence,
                )
                for context_sentence
                in context_sentences
            ]

            sentence_scores.append(
                max(evidence_scores)
            )

        score = (
            sum(sentence_scores)
            / len(sentence_scores)
        )

        return ScoreResult(
            score=score,
            explanation=(
                "Faithfulness was estimated by "
                "matching each answer sentence "
                "against the strongest supporting "
                "sentence in the retrieved context."
            ),
        )