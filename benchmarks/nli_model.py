from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)


@dataclass(frozen=True)
class NLIModelInfo:
    model_name: str
    device: str
    entailment_index: int


class NLIEntailmentModel:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.model_name = model_name

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading NLI tokenizer: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
            )
        )

        print(
            f"Loading NLI model on "
            f"{self.device}..."
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.entailment_index = (
            self._find_entailment_index()
        )

        print(
            "NLI labels:",
            self.model.config.id2label,
        )

        print(
            "Entailment index:",
            self.entailment_index,
        )

    def _find_entailment_index(
        self,
    ) -> int:
        for index, label in (
            self.model.config.id2label.items()
        ):
            clean_label = str(
                label
            ).lower()

            if (
                "entail" in clean_label
                and "not_entail"
                not in clean_label
            ):
                return int(index)

        raise RuntimeError(
            "Could not identify the "
            "entailment label."
        )

    @property
    def info(self) -> NLIModelInfo:
        return NLIModelInfo(
            model_name=self.model_name,
            device=str(
                self.device
            ),
            entailment_index=(
                self.entailment_index
            ),
        )

    def entailment_scores(
        self,
        pairs: list[
            tuple[str, str]
        ],
        batch_size: int = 8,
    ) -> list[float]:
        if not pairs:
            return []

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be "
                "greater than zero."
            )

        scores: list[float] = []

        for start in range(
            0,
            len(pairs),
            batch_size,
        ):
            batch = pairs[
                start:
                start + batch_size
            ]

            premises = [
                premise
                for premise, _
                in batch
            ]

            hypotheses = [
                hypothesis
                for _, hypothesis
                in batch
            ]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for key, value
                in inputs.items()
            }

            with torch.inference_mode():
                logits = self.model(
                    **inputs
                ).logits

                probabilities = (
                    torch.softmax(
                        logits,
                        dim=-1,
                    )
                )

            entailment = (
                probabilities[
                    :,
                    self.entailment_index
                ]
                .detach()
                .cpu()
                .tolist()
            )

            scores.extend(
                float(value)
                for value in entailment
            )

        return scores
