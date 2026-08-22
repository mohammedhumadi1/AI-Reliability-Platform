from app.models import EvaluationMetric
from app.schemas.evaluation_result import (
    EvaluationResultResponse,
)
from evaluation.versioning import (
    EMBEDDING_MODEL_NAME,
    EVALUATION_VERSION,
    SCORE_PROFILE_RAG_BASIC,
    SCORE_PROFILE_RAG_WITH_KB,
    SCORING_VERSION,
    resolve_score_profile,
)


def test_evaluation_metric_has_audit_columns() -> None:
    columns = {
        column.name
        for column in EvaluationMetric.__table__.columns
    }

    assert {
        "evaluation_version",
        "embedding_model_name",
        "scoring_version",
        "score_profile",
        "weights_used",
    } <= columns


def test_score_profile_depends_on_kb_availability() -> None:
    assert (
        resolve_score_profile(None)
        == SCORE_PROFILE_RAG_BASIC
    )
    assert (
        resolve_score_profile(0.0)
        == SCORE_PROFILE_RAG_WITH_KB
    )


def test_evaluation_response_accepts_audit_metadata() -> None:
    response = EvaluationResultResponse(
        correctness_score=1.0,
        faithfulness_score=1.0,
        context_precision_score=1.0,
        answer_relevancy_score=1.0,
        hallucination_risk=0.0,
        status="HEALTHY",
        explanation="Supported.",
        evaluation_version=EVALUATION_VERSION,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        scoring_version=SCORING_VERSION,
        score_profile=SCORE_PROFILE_RAG_BASIC,
        weights_used={"faithfulness": 0.25},
    )
    assert response.evaluation_version == "semantic-v1"
