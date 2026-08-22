from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    __table_args__ = (
        UniqueConstraint(
            "health_check_id",
            name="uq_evaluation_metrics_health_check_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    health_check_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("health_checks.id"),
        nullable=False,
        index=True,
    )

    correctness_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    faithfulness_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    context_precision_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    context_recall_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    answer_relevancy_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    hallucination_risk: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    overall_health_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    health_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    evaluation_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evaluation_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    embedding_model_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    scoring_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    score_profile: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    weights_used: Mapped[
        dict[str, float] | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    health_check: Mapped["HealthCheck"] = relationship(
        back_populates="evaluation_metric",
    )