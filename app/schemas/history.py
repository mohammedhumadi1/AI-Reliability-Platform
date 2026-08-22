from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.evaluation_result import (
    DiagnosisResponse,
    EvaluationResultResponse,
    KnowledgeBaseVerificationResponse,
    RecommendationResponse,
)


class RetrievedContextResponse(BaseModel):
    text: str
    source: str | None = None
    rank: int | None = None

    retrieval_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class HealthCheckHistoryItem(BaseModel):
    health_check_id: uuid.UUID
    project_id: str
    application_version: str | None = None
    status: str

    overall_health_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    health_status: str | None = None
    evaluation_status: str | None = None
    evaluation_version: str | None = None
    scoring_version: str | None = None
    score_profile: str | None = None
    diagnosis_category: str | None = None
    knowledge_base_status: str | None = None

    created_at: datetime


class HealthCheckHistoryResponse(BaseModel):
    total: int = Field(
        ge=0,
    )

    limit: int = Field(
        ge=1,
    )

    offset: int = Field(
        ge=0,
    )

    items: list[HealthCheckHistoryItem]


class HealthCheckDetailResponse(BaseModel):
    health_check_id: uuid.UUID
    project_id: str

    application_version: str | None = None

    status: str

    overall_health_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    health_status: str | None = None

    question: str
    answer: str

    reference_answer: str | None = None
    prompt: str | None = None

    model: dict | None = None
    retriever: dict | None = None
    performance: dict | None = None

    evaluation: EvaluationResultResponse | None = None
    knowledge_base_verification: (
        KnowledgeBaseVerificationResponse
        | None
    ) = None
    diagnosis: DiagnosisResponse | None = None

    contexts: list[RetrievedContextResponse] = Field(
        default_factory=list,
    )

    recommendations: list[
        RecommendationResponse
    ] = Field(
        default_factory=list,
    )

    created_at: datetime