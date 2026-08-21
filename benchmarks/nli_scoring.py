def strict_answer_score(
    claim_scores: list[float],
) -> float:
    """
    Strict faithfulness:
    every claim must be supported.

    The least-supported claim determines
    the answer-level score.
    """
    if not claim_scores:
        return 0.0

    return min(
        float(score)
        for score in claim_scores
    )


def classify_score(
    score: float,
    threshold: float,
) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    return float(score) >= threshold
