from datasets import load_dataset


def load_ragbench_samples(
    subset: str = "covidqa",
    split: str = "validation",
    limit: int = 3,
):
    dataset = load_dataset(
        "galileo-ai/ragbench",
        subset,
        split=split,
    )

    samples = []

    for row in dataset.select(range(min(limit, len(dataset)))):
        samples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "contexts": row["documents"],
                "response": row["response"],
                "adherence_score": row["adherence_score"],
                "relevance_score": row["relevance_score"],
                "utilization_score": row["utilization_score"],
                "completeness_score": row["completeness_score"],
            }
        )

    return samples


if __name__ == "__main__":
    samples = load_ragbench_samples(limit=3)

    for i, sample in enumerate(samples, start=1):
        print("=" * 80)
        print(f"Sample {i}")
        print("ID:", sample["id"])
        print("Question:", sample["question"])
        print("Response:", sample["response"])
        print("Documents:", len(sample["contexts"]))
        print("Adherence:", sample["adherence_score"])
        print("Relevance:", sample["relevance_score"])
        print("Utilization:", sample["utilization_score"])
        print("Completeness:", sample["completeness_score"])
