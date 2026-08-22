from types import SimpleNamespace

from app.routers.history import (
    build_evaluation_response,
)
from app.schemas.history import (
    HealthCheckHistoryItem,
)


def test_history_evaluation_exposes_audit_metadata() -> None:
    metric = SimpleNamespace(
        correctness_score=1.0,
        faithfulness_score=1.0,
        context_precision_score=1.0,
        context_recall_score=1.0,
        answer_relevancy_score=1.0,
        hallucination_risk=0.0,
        evaluation_status="HEALTHY",
        explanation="Supported.",
        evaluation_version="semantic-v1",
        embedding_model_name="embedding-model",
        scoring_version="health-score-v1",
        score_profile="RAG_WITH_KB",
        weights_used={"faithfulness": 0.25},
    )
    health_check = SimpleNamespace(
        evaluation_metric=metric,
    )

    response = build_evaluation_response(
        health_check
    )

    assert response is not None
    assert response.evaluation_version == "semantic-v1"
    assert response.embedding_model_name == "embedding-model"
    assert response.scoring_version == "health-score-v1"
    assert response.score_profile == "RAG_WITH_KB"
    assert response.weights_used == {
        "faithfulness": 0.25
    }


def test_history_item_accepts_audit_summary() -> None:
    item = HealthCheckHistoryItem(
        health_check_id="11111111-1111-1111-1111-111111111111",
        project_id="project-a",
        status="COMPLETED",
        evaluation_version="semantic-v1",
        scoring_version="health-score-v1",
        score_profile="RAG_BASIC",
        created_at="2026-08-23T00:00:00",
    )

    assert item.evaluation_version == "semantic-v1"
    assert item.scoring_version == "health-score-v1"
    assert item.score_profile == "RAG_BASIC"
