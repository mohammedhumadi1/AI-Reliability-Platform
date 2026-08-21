from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RAGBenchSample:
    sample_id: str
    question: str
    contexts: list[str]
    response: str
    supported: bool


def select_balanced_indices(
    labels: list[bool],
    limit: int,
) -> list[int]:
    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero."
        )

    if limit > len(labels):
        raise ValueError(
            "limit cannot exceed the available samples."
        )

    positive_target = limit // 2
    negative_target = limit - positive_target

    positive_indices = [
        index
        for index, label in enumerate(labels)
        if label
    ]

    negative_indices = [
        index
        for index, label in enumerate(labels)
        if not label
    ]

    if len(positive_indices) < positive_target:
        raise ValueError(
            "Not enough supported samples for a balanced selection."
        )

    if len(negative_indices) < negative_target:
        raise ValueError(
            "Not enough unsupported samples for a balanced selection."
        )

    selected = (
        positive_indices[:positive_target]
        + negative_indices[:negative_target]
    )

    return sorted(selected)


def _row_to_sample(
    row: dict[str, Any],
) -> RAGBenchSample:
    documents = row.get("documents") or []

    contexts = [
        str(document).strip()
        for document in documents
        if str(document).strip()
    ]

    return RAGBenchSample(
        sample_id=str(row["id"]),
        question=str(
            row["question"]
        ).strip(),
        contexts=contexts,
        response=str(
            row["response"]
        ).strip(),
        supported=bool(
            row["adherence_score"]
        ),
    )


def load_ragbench_samples(
    subset: str = "covidqa",
    split: str = "validation",
    limit: int | None = None,
    balanced: bool = False,
) -> list[RAGBenchSample]:
    from datasets import load_dataset

    dataset = load_dataset(
        "galileo-ai/ragbench",
        subset,
        split=split,
    )

    if limit is None:
        indices = list(
            range(len(dataset))
        )

    elif balanced:
        labels = [
            bool(row["adherence_score"])
            for row in dataset
        ]

        indices = select_balanced_indices(
            labels,
            limit,
        )

    else:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        indices = list(
            range(
                min(
                    limit,
                    len(dataset),
                )
            )
        )

    return [
        _row_to_sample(
            dataset[index]
        )
        for index in indices
    ]
