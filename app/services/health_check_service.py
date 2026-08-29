from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models import (
    Diagnosis,
    EvaluationMetric,
    HealthCheck,
    KnowledgeBaseVerification,
    RecommendationRecord,
    RetrievedContext,
)
from app.schemas.evaluation_result import (
    DiagnosisResponse,
    EvaluationResultResponse,
    HealthCheckResponse,
    KnowledgeBaseVerificationResponse,
    RecommendationResponse,
)
from app.schemas.health_check import HealthCheckRequest
from evaluation.pipeline import (
    EvaluationPipelineResult,
    run_evaluation,
)
from evaluation.versioning import (
    EMBEDDING_MODEL_NAME,
    EVALUATION_VERSION,
    SCORING_VERSION,
    resolve_score_profile,
)
from knowledge_base.verification_agent import (
    VerificationResult,
    verify_answer,
)
from recommendation_engine.engine import (
    generate_recommendations,
)
from reporting.health_score import (
    calculate_health_score,
)
from root_cause.prompt_evidence import (
    analyze_prompt_evidence,
)
from root_cause.rules.pipeline import (
    run_rules_pipeline,
)


def build_verification_response(
    verification: VerificationResult,
) -> KnowledgeBaseVerificationResponse:
    return KnowledgeBaseVerificationResponse(
        status=verification.status,
        evidence_found=verification.evidence_found,
        is_supported=verification.is_supported,
        best_match_text=(
            verification.best_match_text
        ),
        best_match_source=(
            verification.best_match_source
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


def build_root_cause_metrics(
    result: EvaluationPipelineResult,
    verification: VerificationResult,
    prompt: str | None,
) -> dict:
    """Build evidence-aware metrics for root-cause diagnosis."""
    metrics = {
        "context_precision": (
            result.context_precision_score
        ),
        "faithfulness": (
            result.faithfulness_score
        ),
        "answer_relevancy": (
            result.answer_relevancy_score
        ),
        "verification_status": (
            verification.status
        ),
        "context_alignment_score": (
            verification.context_alignment_score
        ),
        "verification_explanation": (
            verification.explanation
        ),
    }

    if result.context_recall_score is not None:
        metrics["context_recall"] = (
            result.context_recall_score
        )

    prompt_evidence = analyze_prompt_evidence(
        prompt
    )

    if prompt_evidence is not None:
        metrics["prompt_evidence"] = (
            prompt_evidence
        )

    return metrics


def execute_health_check(
    payload: HealthCheckRequest,
    db: Session,
) -> HealthCheckResponse:
    context_texts = [
        context.text
        for context in payload.contexts
    ]

    # 1) Existing RAG evaluation.
    result = run_evaluation(
        question=payload.question,
        answer=payload.answer,
        contexts=context_texts,
        reference_answer=(
            payload.reference_answer
        ),
    )

    # 2) Independent company-KB verification.
    verification = verify_answer(
        project_id=payload.project_id,
        question=payload.question,
        answer=payload.answer,
        rag_contexts=context_texts,
    )

    # 3) Root cause. Direct KB evidence has
    # priority when it is available.
    root_cause_metrics = build_root_cause_metrics(
        result=result,
        verification=verification,
        prompt=payload.prompt,
    )

    diagnosis_dict = run_rules_pipeline(
        root_cause_metrics
    )

    # 4) Overall health score. KB support
    # is optional, so projects without an
    # indexed KB preserve previous behavior.
    health_metrics = {
        "faithfulness": (
            result.faithfulness_score
        ),
        "answer_relevancy": (
            result.answer_relevancy_score
        ),
        "answer_correctness": (
            result.correctness_score
        ),
        "context_precision": (
            result.context_precision_score
        ),
        "context_recall": (
            result.context_recall_score
        ),
        "knowledge_base_support": (
            verification.health_score_component
        ),
    }

    health_score = calculate_health_score(
        health_metrics
    )

    score_profile = resolve_score_profile(
        verification.health_score_component
    )

    # 5) Recommendation engine.
    recommendations = (
        generate_recommendations(
            diagnosis_dict
        )
    )

    # 6) Persist one atomic PostgreSQL
    # health-check record and its children.
    db_health_check = HealthCheck(
        id=uuid.uuid4(),
        project_id=payload.project_id,
        application_version=(
            payload.application_version
        ),
        question=payload.question,
        answer=payload.answer,
        reference_answer=(
            payload.reference_answer
        ),
        prompt=payload.prompt,
        model_config_data=(
            payload.model.model_dump()
        ),
        retriever_config=(
            payload.retriever.model_dump()
            if payload.retriever
            else None
        ),
        performance=(
            payload.performance.model_dump()
            if payload.performance
            else None
        ),
        status="COMPLETED",
    )

    diagnosis_response = None

    try:
        db.add(db_health_check)
        db.flush()

        for context in payload.contexts:
            db.add(
                RetrievedContext(
                    id=uuid.uuid4(),
                    health_check_id=(
                        db_health_check.id
                    ),
                    text=context.text,
                    source=context.source,
                    rank=context.rank,
                    retrieval_score=(
                        context.retrieval_score
                    ),
                )
            )

        db.add(
            EvaluationMetric(
                id=uuid.uuid4(),
                health_check_id=(
                    db_health_check.id
                ),
                correctness_score=(
                    result.correctness_score
                ),
                faithfulness_score=(
                    result.faithfulness_score
                ),
                context_precision_score=(
                    result.context_precision_score
                ),
                context_recall_score=(
                    result.context_recall_score
                ),
                answer_relevancy_score=(
                    result.answer_relevancy_score
                ),
                hallucination_risk=(
                    result.hallucination_risk
                ),
                overall_health_score=(
                    health_score.score
                ),
                health_status=(
                    health_score.status
                ),
                evaluation_status=(
                    result.status
                ),
                explanation=(
                    result.explanation
                ),
                evaluation_version=EVALUATION_VERSION,
                embedding_model_name=EMBEDDING_MODEL_NAME,
                scoring_version=SCORING_VERSION,
                score_profile=score_profile,
                weights_used=health_score.weights_used,
            )
        )

        db.add(
            KnowledgeBaseVerification(
                id=uuid.uuid4(),
                health_check_id=(
                    db_health_check.id
                ),
                status=verification.status,
                evidence_found=(
                    verification.evidence_found
                ),
                is_supported=(
                    verification.is_supported
                ),
                best_match_text=(
                    verification.best_match_text
                    or None
                ),
                best_match_source=(
                    verification.best_match_source
                    or None
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
                explanation=(
                    verification.explanation
                ),
            )
        )

        if diagnosis_dict:
            db.add(
                Diagnosis(
                    id=uuid.uuid4(),
                    health_check_id=(
                        db_health_check.id
                    ),
                    category=(
                        diagnosis_dict[
                            "category"
                        ]
                    ),
                    subcategory=(
                        diagnosis_dict.get(
                            "subcategory"
                        )
                    ),
                    severity=(
                        diagnosis_dict[
                            "severity"
                        ]
                    ),
                    confidence=(
                        diagnosis_dict[
                            "confidence"
                        ]
                    ),
                    explanation=(
                        diagnosis_dict[
                            "explanation"
                        ]
                    ),
                )
            )

            diagnosis_response = (
                DiagnosisResponse(
                    **diagnosis_dict
                )
            )

        for item in recommendations:
            db.add(
                RecommendationRecord(
                    id=uuid.uuid4(),
                    health_check_id=(
                        db_health_check.id
                    ),
                    priority=item.priority,
                    action=item.action,
                    expected_impact=(
                        item.expected_impact
                    ),
                    difficulty=(
                        item.difficulty
                    ),
                    affected_component=(
                        item.affected_component
                    ),
                    supporting_evidence=(
                        item.supporting_evidence
                    ),
                )
            )

        db.commit()

    except Exception:
        db.rollback()
        raise

    recommendation_responses = [
        RecommendationResponse(
            priority=item.priority,
            action=item.action,
            expected_impact=(
                item.expected_impact
            ),
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

    return HealthCheckResponse(
        health_check_id=(
            db_health_check.id
        ),
        project_id=payload.project_id,
        status="COMPLETED",
        overall_health_score=(
            health_score.score
        ),
        health_status=(
            health_score.status
        ),
        question=payload.question,
        answer=payload.answer,
        evaluation=(
            EvaluationResultResponse(
                correctness_score=(
                    result.correctness_score
                ),
                faithfulness_score=(
                    result.faithfulness_score
                ),
                context_precision_score=(
                    result.context_precision_score
                ),
                context_recall_score=(
                    result.context_recall_score
                ),
                answer_relevancy_score=(
                    result.answer_relevancy_score
                ),
                hallucination_risk=(
                    result.hallucination_risk
                ),
                status=result.status,
                explanation=(
                    result.explanation
                ),
                evaluation_version=EVALUATION_VERSION,
                embedding_model_name=EMBEDDING_MODEL_NAME,
                scoring_version=SCORING_VERSION,
                score_profile=score_profile,
                weights_used=health_score.weights_used,
            )
        ),
        knowledge_base_verification=(
            build_verification_response(
                verification
            )
        ),
        diagnosis=diagnosis_response,
        recommendations=(
            recommendation_responses
        ),
    )
