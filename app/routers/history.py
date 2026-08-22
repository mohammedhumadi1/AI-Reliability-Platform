from __future__ import annotations

import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy import func, select
from sqlalchemy.orm import (
    Session,
    selectinload,
)

from app.database import get_db
from app.models import HealthCheck
from app.schemas.evaluation_result import (
    DiagnosisResponse,
    EvaluationResultResponse,
    KnowledgeBaseVerificationResponse,
    RecommendationResponse,
)
from app.schemas.history import (
    HealthCheckDetailResponse,
    HealthCheckHistoryItem,
    HealthCheckHistoryResponse,
    RetrievedContextResponse,
)


router = APIRouter(
    prefix="/api/v1/health-checks",
    tags=["Health Check History"],
)


def build_evaluation_response(
    health_check: HealthCheck,
) -> EvaluationResultResponse | None:
    metric = health_check.evaluation_metric

    if metric is None:
        return None

    return EvaluationResultResponse(
        correctness_score=metric.correctness_score,
        faithfulness_score=metric.faithfulness_score,
        context_precision_score=(
            metric.context_precision_score
        ),
        context_recall_score=(
            metric.context_recall_score
        ),
        answer_relevancy_score=(
            metric.answer_relevancy_score
        ),
        hallucination_risk=metric.hallucination_risk,
        status=metric.evaluation_status,
        explanation=metric.explanation,
        evaluation_version=metric.evaluation_version,
        embedding_model_name=metric.embedding_model_name,
        scoring_version=metric.scoring_version,
        score_profile=metric.score_profile,
        weights_used=metric.weights_used,
    )


def build_knowledge_base_verification_response(
    health_check: HealthCheck,
) -> KnowledgeBaseVerificationResponse | None:
    verification = (
        health_check.knowledge_base_verification
    )

    if verification is None:
        return None

    return KnowledgeBaseVerificationResponse(
        status=verification.status,
        evidence_found=(
            verification.evidence_found
        ),
        is_supported=(
            verification.is_supported
        ),
        best_match_text=(
            verification.best_match_text
            or ""
        ),
        best_match_source=(
            verification.best_match_source
            or ""
        ),
        similarity_distance=(
            verification.similarity_distance
        ),
        question_relevance_score=(
            verification.question_relevance_score
        ),
        answer_support_score=(
            verification.answer_support_score
        ),
        context_alignment_score=(
            verification.context_alignment_score
        ),
        numeric_contradiction=(
            verification.numeric_contradiction
        ),
        explanation=verification.explanation,
    )


def build_diagnosis_response(
    health_check: HealthCheck,
) -> DiagnosisResponse | None:
    if not health_check.diagnoses:
        return None

    diagnosis = sorted(
        health_check.diagnoses,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]

    return DiagnosisResponse(
        category=diagnosis.category,
        subcategory=diagnosis.subcategory,
        severity=diagnosis.severity,
        confidence=diagnosis.confidence,
        explanation=diagnosis.explanation,
    )


def build_recommendation_responses(
    health_check: HealthCheck,
) -> list[RecommendationResponse]:
    recommendations = sorted(
        health_check.recommendations,
        key=lambda item: item.priority,
    )

    return [
        RecommendationResponse(
            priority=item.priority,
            action=item.action,
            expected_impact=item.expected_impact,
            difficulty=item.difficulty,
            affected_component=(
                item.affected_component
            ),
            supporting_evidence=(
                item.supporting_evidence
            ),
        )
        for item in recommendations
    ]


def build_context_responses(
    health_check: HealthCheck,
) -> list[RetrievedContextResponse]:
    contexts = sorted(
        health_check.contexts,
        key=lambda item: (
            item.rank is None,
            item.rank if item.rank is not None else 0,
            item.created_at,
        ),
    )

    return [
        RetrievedContextResponse(
            text=item.text,
            source=item.source,
            rank=item.rank,
            retrieval_score=item.retrieval_score,
        )
        for item in contexts
    ]


@router.get(
    "",
    response_model=HealthCheckHistoryResponse,
)
def list_health_checks(
    project_id: str | None = Query(
        default=None,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    db: Session = Depends(get_db),
) -> HealthCheckHistoryResponse:
    filters = []

    if project_id:
        filters.append(
            HealthCheck.project_id == project_id
        )

    count_statement = (
        select(func.count())
        .select_from(HealthCheck)
    )

    if filters:
        count_statement = count_statement.where(
            *filters
        )

    total = db.execute(
        count_statement
    ).scalar_one()

    statement = (
        select(HealthCheck)
        .options(
            selectinload(
                HealthCheck.evaluation_metric
            ),
            selectinload(
                HealthCheck.diagnoses
            ),
            selectinload(
                HealthCheck.knowledge_base_verification
            ),
        )
        .order_by(
            HealthCheck.created_at.desc()
        )
        .offset(offset)
        .limit(limit)
    )

    if filters:
        statement = statement.where(
            *filters
        )

    health_checks = (
        db.execute(statement)
        .scalars()
        .all()
    )

    items: list[HealthCheckHistoryItem] = []

    for health_check in health_checks:
        metric = health_check.evaluation_metric
        verification = (
            health_check.knowledge_base_verification
        )

        diagnosis_category = None

        if health_check.diagnoses:
            latest_diagnosis = sorted(
                health_check.diagnoses,
                key=lambda item: item.created_at,
                reverse=True,
            )[0]

            diagnosis_category = (
                latest_diagnosis.category
            )

        items.append(
            HealthCheckHistoryItem(
                health_check_id=health_check.id,
                project_id=health_check.project_id,
                application_version=(
                    health_check.application_version
                ),
                status=health_check.status,
                overall_health_score=(
                    metric.overall_health_score
                    if metric
                    else None
                ),
                health_status=(
                    metric.health_status
                    if metric
                    else None
                ),
                evaluation_status=(
                    metric.evaluation_status
                    if metric
                    else None
                ),
                diagnosis_category=(
                    diagnosis_category
                ),
                knowledge_base_status=(
                    verification.status
                    if verification
                    else None
                ),
                created_at=health_check.created_at,
                evaluation_version=(
                    metric.evaluation_version
                    if metric
                    else None
                ),
                scoring_version=(
                    metric.scoring_version
                    if metric
                    else None
                ),
                score_profile=(
                    metric.score_profile
                    if metric
                    else None
                ),
            )
        )

    return HealthCheckHistoryResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


@router.get(
    "/{health_check_id}",
    response_model=HealthCheckDetailResponse,
)
def get_health_check(
    health_check_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> HealthCheckDetailResponse:
    statement = (
        select(HealthCheck)
        .options(
            selectinload(
                HealthCheck.evaluation_metric
            ),
            selectinload(
                HealthCheck.diagnoses
            ),
            selectinload(
                HealthCheck.contexts
            ),
            selectinload(
                HealthCheck.recommendations
            ),
            selectinload(
                HealthCheck.knowledge_base_verification
            ),
        )
        .where(
            HealthCheck.id == health_check_id
        )
    )

    health_check = (
        db.execute(statement)
        .scalars()
        .one_or_none()
    )

    if health_check is None:
        raise HTTPException(
            status_code=404,
            detail="Health check not found",
        )

    metric = health_check.evaluation_metric

    return HealthCheckDetailResponse(
        health_check_id=health_check.id,
        project_id=health_check.project_id,
        application_version=(
            health_check.application_version
        ),
        status=health_check.status,
        overall_health_score=(
            metric.overall_health_score
            if metric
            else None
        ),
        health_status=(
            metric.health_status
            if metric
            else None
        ),
        question=health_check.question,
        answer=health_check.answer,
        reference_answer=(
            health_check.reference_answer
        ),
        prompt=health_check.prompt,
        model=health_check.model_config_data,
        retriever=health_check.retriever_config,
        performance=health_check.performance,
        evaluation=build_evaluation_response(
            health_check
        ),
        knowledge_base_verification=(
            build_knowledge_base_verification_response(
                health_check
            )
        ),
        diagnosis=build_diagnosis_response(
            health_check
        ),
        contexts=build_context_responses(
            health_check
        ),
        recommendations=(
            build_recommendation_responses(
                health_check
            )
        ),
        created_at=health_check.created_at,
    )
