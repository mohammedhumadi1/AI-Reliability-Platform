from evaluation.generation.embedding_service import (
    MODEL_NAME,
)


EVALUATION_VERSION = "semantic-v1"
EMBEDDING_MODEL_NAME = MODEL_NAME
SCORING_VERSION = "health-score-v1"

SCORE_PROFILE_RAG_BASIC = "RAG_BASIC"
SCORE_PROFILE_RAG_WITH_KB = "RAG_WITH_KB"


def resolve_score_profile(
    knowledge_base_support: float | None,
) -> str:
    if knowledge_base_support is None:
        return SCORE_PROFILE_RAG_BASIC

    return SCORE_PROFILE_RAG_WITH_KB
