from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float
    label: str


@lru_cache(maxsize=1)
def _load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.eval()

    return tokenizer, model


def _result_from_probabilities(
    probabilities,
    model,
) -> NLIResult:
    scores = {}

    for index, probability in enumerate(
        probabilities
    ):
        label = str(
            model.config.id2label[index]
        ).lower()

        scores[label] = float(
            probability.item()
        )

    entailment = scores.get(
        "entailment",
        0.0,
    )

    neutral = scores.get(
        "neutral",
        0.0,
    )

    contradiction = scores.get(
        "contradiction",
        0.0,
    )

    label_scores = {
        "entailment": entailment,
        "neutral": neutral,
        "contradiction": contradiction,
    }

    best_label = max(
        label_scores,
        key=label_scores.get,
    )

    return NLIResult(
        entailment=round(entailment, 4),
        neutral=round(neutral, 4),
        contradiction=round(
            contradiction,
            4,
        ),
        label=best_label,
    )


def evaluate_nli(
    premise: str,
    hypothesis: str,
) -> NLIResult:
    return evaluate_nli_batch(
        [(premise, hypothesis)]
    )[0]


def evaluate_nli_batch(
    pairs: list[tuple[str, str]],
    batch_size: int = 16,
) -> list[NLIResult]:
    if not pairs:
        return []

    cleaned_pairs = []

    for premise, hypothesis in pairs:
        clean_premise = premise.strip()
        clean_hypothesis = hypothesis.strip()

        if not clean_premise:
            raise ValueError(
                "NLI premise cannot be empty."
            )

        if not clean_hypothesis:
            raise ValueError(
                "NLI hypothesis cannot be empty."
            )

        cleaned_pairs.append(
            (
                clean_premise,
                clean_hypothesis,
            )
        )

    tokenizer, model = _load_nli_model()

    results = []

    for start in range(
        0,
        len(cleaned_pairs),
        batch_size,
    ):
        batch = cleaned_pairs[
            start:start + batch_size
        ]

        premises = [
            pair[0]
            for pair in batch
        ]

        hypotheses = [
            pair[1]
            for pair in batch
        ]

        inputs = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = model(
                **inputs
            ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        for row in probabilities:
            results.append(
                _result_from_probabilities(
                    row,
                    model,
                )
            )

    return results