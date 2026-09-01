from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from api_client import (
    APIClientError,
    create_health_check,
    get_health,
    get_health_check,
    get_history,
    upload_knowledge_base_document,
    verify_knowledge_base_answer,
)

from i18n import (
    localize_action,
    localize_diagnosis_explanation,
    localize_evaluation_explanation,
    localize_kb_explanation,
    localize_value,
    tr,
)


DEFAULT_API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8002",
)


st.set_page_config(
    page_title=(
        "AI Reliability Platform | "
        "منصة موثوقية الذكاء الاصطناعي"
    ),
    layout="wide",
)


def apply_direction(
    lang: str,
) -> None:
    direction = (
        "rtl"
        if lang == "ar"
        else "ltr"
    )

    align = (
        "right"
        if lang == "ar"
        else "left"
    )

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            direction: {direction};
        }}

        [data-testid="stSidebar"] {{
            direction: {direction};
        }}

        [data-testid="stMarkdownContainer"] {{
            text-align: {align};
        }}

        [data-testid="stMetricLabel"] {{
            text-align: {align};
        }}

        [data-testid="stMetricValue"] {{
            direction: ltr;
            text-align: {align};
        }}

        input, textarea {{
            direction: auto !important;
            text-align: start !important;
        }}

        .stDataFrame {{
            direction: {direction};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def percent(
    value: float | None,
    lang: str,
) -> str:
    if value is None:
        return tr(
            lang,
            "not_available",
        )

    return f"{value * 100:.1f}%"


def score_value(
    value: int | None,
    lang: str,
) -> str:
    if value is None:
        return tr(
            lang,
            "not_available",
        )

    return f"{value}/100"


def format_datetime(
    value: str | None,
    lang: str,
) -> str:
    if not value:
        return tr(
            lang,
            "not_available",
        )

    try:
        parsed = datetime.fromisoformat(
            value
        )

        return parsed.strftime(
            "%Y-%m-%d %H:%M"
        )

    except ValueError:
        return value


def render_status_banner(
    status: str | None,
    lang: str,
) -> None:
    if not status:
        return

    label = localize_value(
        status,
        lang,
    )

    message = (
        f"{tr(lang, 'health_status')}: "
        f"{label}"
    )

    if status in {
        "EXCELLENT",
        "GOOD",
        "HEALTHY",
    }:
        st.success(message)

    elif status == "WARNING":
        st.warning(message)

    else:
        st.error(message)


def render_evaluation(
    evaluation: dict | None,
    lang: str,
) -> None:
    st.subheader(
        tr(
            lang,
            "evaluation_metrics",
        )
    )

    if not evaluation:
        st.info(
            tr(
                lang,
                "no_evaluation",
            )
        )
        return

    row1 = st.columns(3)

    row1[0].metric(
        tr(lang, "correctness"),
        percent(
            evaluation.get(
                "correctness_score"
            ),
            lang,
        ),
        border=True,
    )

    row1[1].metric(
        tr(lang, "faithfulness"),
        percent(
            evaluation.get(
                "faithfulness_score"
            ),
            lang,
        ),
        border=True,
    )

    row1[2].metric(
        tr(lang, "context_precision"),
        percent(
            evaluation.get(
                "context_precision_score"
            ),
            lang,
        ),
        border=True,
    )

    row2 = st.columns(3)

    row2[0].metric(
        tr(lang, "context_recall"),
        percent(
            evaluation.get(
                "context_recall_score"
            ),
            lang,
        ),
        border=True,
    )

    row2[1].metric(
        tr(lang, "answer_relevancy"),
        percent(
            evaluation.get(
                "answer_relevancy_score"
            ),
            lang,
        ),
        border=True,
    )

    row2[2].metric(
        tr(lang, "hallucination_risk"),
        percent(
            evaluation.get(
                "hallucination_risk"
            ),
            lang,
        ),
        border=True,
    )

    evaluation_status = (
        evaluation.get(
            "status"
        )
    )

    if evaluation_status:
        st.write(
            f"**{tr(lang, 'evaluation_status')}:** "
            f"`{localize_value(evaluation_status, lang)}`"
        )

    explanation = (
        localize_evaluation_explanation(
            evaluation,
            lang,
        )
    )

    if explanation:
        with st.expander(
            tr(
                lang,
                "evaluation_explanation",
            )
        ):
            st.write(
                explanation
            )

            if (
                lang == "ar"
                and evaluation.get(
                    "explanation"
                )
            ):
                st.divider()

                st.caption(
                    tr(
                        lang,
                        "original_explanation",
                    )
                )

                st.write(
                    evaluation[
                        "explanation"
                    ]
                )



def render_knowledge_base_verification(
    verification: dict | None,
    lang: str,
) -> None:
    st.subheader(tr(lang, "knowledge_base_verification"))

    if not verification:
        st.info(tr(lang, "no_kb_verification"))
        return

    status = verification.get("status")
    status_label = localize_value(status, lang)

    message_map = {
        "SUPPORTED": "supported_message",
        "CONTRADICTED": "contradicted_message",
        "UNSUPPORTED": "unsupported_message",
        "NO_RELEVANT_EVIDENCE": "no_relevant_evidence_message",
        "NOT_AVAILABLE": "kb_not_available_message",
    }
    message = tr(lang, message_map.get(status, "kb_verification_status"))

    if status == "SUPPORTED":
        st.success(message)
    elif status in {"CONTRADICTED", "UNSUPPORTED"}:
        st.error(message)
    elif status == "NO_RELEVANT_EVIDENCE":
        st.warning(message)
    else:
        st.info(message)

    st.write(f"**{tr(lang, 'kb_verification_status')}:** `{status_label}`")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        tr(lang, "answer_support"),
        percent(verification.get("answer_support_score"), lang),
        border=True,
    )
    col2.metric(
        tr(lang, "question_relevance"),
        percent(verification.get("question_relevance_score"), lang),
        border=True,
    )
    col3.metric(
        tr(lang, "context_alignment"),
        percent(verification.get("context_alignment_score"), lang),
        border=True,
    )

    evidence_found = verification.get("evidence_found", False)
    st.write(
        f"**{tr(lang, 'evidence_found')}:** "
        f"{tr(lang, 'yes') if evidence_found else tr(lang, 'no')}"
    )

    source = verification.get("best_match_source")
    if source:
        st.write(f"**{tr(lang, 'source')}:** {source}")

    distance = verification.get("similarity_distance")
    if distance is not None:
        st.write(f"**{tr(lang, 'similarity_distance')}:** {distance:.4f}")

    explanation = localize_kb_explanation(verification, lang)
    if explanation:
        st.write(f"**{tr(lang, 'explanation')}:**")
        st.write(explanation)

    best_text = verification.get("best_match_text")
    if best_text:
        with st.expander(tr(lang, "best_matching_text")):
            st.write(best_text)


def render_diagnosis(
    diagnosis: dict | None,
    evaluation: dict | None,
    lang: str,
) -> None:
    st.subheader(
        tr(
            lang,
            "root_cause",
        )
    )

    if not diagnosis:
        st.info(
            tr(
                lang,
                "no_diagnosis",
            )
        )
        return

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        tr(lang, "category"),
        localize_value(
            diagnosis.get(
                "category"
            ),
            lang,
        ),
        border=True,
    )

    col2.metric(
        tr(lang, "severity"),
        localize_value(
            diagnosis.get(
                "severity"
            ),
            lang,
        ),
        border=True,
    )

    col3.metric(
        tr(lang, "confidence"),
        percent(
            diagnosis.get(
                "confidence"
            ),
            lang,
        ),
        border=True,
    )

    subcategory = diagnosis.get(
        "subcategory"
    )

    if subcategory:
        st.write(
            f"**{tr(lang, 'subcategory')}:** "
            f"`{localize_value(subcategory, lang)}`"
        )

    explanation = (
        localize_diagnosis_explanation(
            diagnosis,
            evaluation,
            lang,
        )
    )

    if explanation:
        st.write(
            explanation
        )


def render_recommendations(
    recommendations: list[dict],
    diagnosis: dict | None,
    evaluation: dict | None,
    lang: str,
) -> None:
    st.subheader(
        tr(
            lang,
            "recommendations",
        )
    )

    if not recommendations:
        st.info(
            tr(
                lang,
                "no_recommendations",
            )
        )
        return

    recommendations = sorted(
        recommendations,
        key=lambda item: item.get(
            "priority",
            999,
        ),
    )

    localized_evidence = (
        localize_diagnosis_explanation(
            diagnosis,
            evaluation,
            lang,
        )
    )

    for item in recommendations:
        priority = item.get(
            "priority",
            "?",
        )

        action = localize_action(
            item.get(
                "action",
                "",
            ),
            lang,
        )

        title = (
            f"{tr(lang, 'priority')} "
            f"{priority} — {action}"
        )

        with st.expander(
            title,
            expanded=(
                priority == 1
            ),
        ):
            st.write(
                f"**{tr(lang, 'expected_impact')}:** "
                f"{localize_value(item.get('expected_impact'), lang)}"
            )

            st.write(
                f"**{tr(lang, 'difficulty')}:** "
                f"{localize_value(item.get('difficulty'), lang)}"
            )

            st.write(
                f"**{tr(lang, 'affected_component')}:** "
                f"{localize_value(item.get('affected_component'), lang)}"
            )

            evidence = (
                localized_evidence
                if lang == "ar"
                else item.get(
                    "supporting_evidence"
                )
            )

            if evidence:
                st.write(
                    f"**{tr(lang, 'supporting_evidence')}:**"
                )

                st.write(
                    evidence
                )


def render_contexts(
    contexts: list[dict],
    lang: str,
) -> None:
    st.subheader(
        tr(
            lang,
            "retrieved_contexts",
        )
    )

    if not contexts:
        st.info(
            tr(
                lang,
                "no_contexts",
            )
        )
        return

    for index, context in enumerate(
        contexts,
        start=1,
    ):
        source = (
            context.get(
                "source"
            )
            or tr(
                lang,
                "not_available",
            )
        )

        rank = (
            context.get(
                "rank"
            )
            or index
        )

        title = (
            f"{tr(lang, 'context')} "
            f"#{rank} — {source}"
        )

        with st.expander(title):
            st.write(
                context.get(
                    "text",
                    "",
                )
            )

            retrieval_score = (
                context.get(
                    "retrieval_score"
                )
            )

            if (
                retrieval_score
                is not None
            ):
                st.write(
                    f"**{tr(lang, 'retrieval_score')}:** "
                    f"{retrieval_score:.4f}"
                )


def render_health_check_detail(
    detail: dict,
    lang: str,
) -> None:
    st.divider()

    top1, top2, top3 = (
        st.columns(
            [1, 1, 2]
        )
    )

    top1.metric(
        tr(
            lang,
            "overall_health_score",
        ),
        score_value(
            detail.get(
                "overall_health_score"
            ),
            lang,
        ),
        border=True,
    )

    top2.metric(
        tr(
            lang,
            "health_status",
        ),
        localize_value(
            detail.get(
                "health_status"
            ),
            lang,
        ),
        border=True,
    )

    top3.metric(
        tr(
            lang,
            "project",
        ),
        detail.get(
            "project_id",
            tr(
                lang,
                "not_available",
            ),
        ),
        border=True,
    )

    render_status_banner(
        detail.get(
            "health_status"
        ),
        lang,
    )

    st.subheader(
        tr(
            lang,
            "question_answer",
        )
    )

    st.write(
        f"**{tr(lang, 'question')}**"
    )

    st.write(
        detail.get(
            "question",
            "",
        )
    )

    st.write(
        f"**{tr(lang, 'generated_answer')}**"
    )

    st.write(
        detail.get(
            "answer",
            "",
        )
    )

    reference_answer = (
        detail.get(
            "reference_answer"
        )
    )

    if reference_answer:
        st.write(
            f"**{tr(lang, 'reference_answer')}**"
        )

        st.write(
            reference_answer
        )

    evaluation = detail.get(
        "evaluation"
    )

    diagnosis = detail.get(
        "diagnosis"
    )

    render_evaluation(
        evaluation,
        lang,
    )

    render_knowledge_base_verification(
        detail.get(
            "knowledge_base_verification"
        ),
        lang,
    )

    render_diagnosis(
        diagnosis,
        evaluation,
        lang,
    )

    render_recommendations(
        detail.get(
            "recommendations",
            [],
        ),
        diagnosis,
        evaluation,
        lang,
    )

    render_contexts(
        detail.get(
            "contexts",
            [],
        ),
        lang,
    )


def load_history(
    api_url: str,
) -> dict:
    try:
        return get_history(
            api_url,
            limit=200,
        )

    except APIClientError as exc:
        st.error(
            str(exc)
        )

        return {
            "total": 0,
            "items": [],
        }


# ============================================================
# Language selector
# ============================================================

stored_lang = st.session_state.get(
    "ui_language",
    "en",
)

language_choice = (
    st.sidebar.segmented_control(
        "Language / اللغة",
        options=[
            "English",
            "العربية",
        ],
        default=(
            "العربية"
            if stored_lang == "ar"
            else "English"
        ),
        selection_mode="single",
    )
)

lang = (
    "ar"
    if language_choice == "العربية"
    else "en"
)

st.session_state[
    "ui_language"
] = lang

apply_direction(lang)


# ============================================================
# Header
# ============================================================

st.title(
    tr(
        lang,
        "platform_title",
    )
)

st.caption(
    tr(
        lang,
        "subtitle",
    )
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header(
        tr(
            lang,
            "connection",
        )
    )

    api_url = st.text_input(
        tr(
            lang,
            "fastapi_url",
        ),
        value=DEFAULT_API_URL,
    )

    if st.button(
        tr(
            lang,
            "check_api",
        ),
        use_container_width=True,
    ):
        try:
            health = get_health(
                api_url
            )

            st.success(
                tr(
                    lang,
                    "connected",
                )
            )

        except APIClientError as exc:
            st.error(
                str(exc)
            )

    st.divider()

    st.caption(
        tr(
            lang,
            "backend",
        )
    )

    st.caption(
        tr(
            lang,
            "dashboard",
        )
    )


# ============================================================
# Tabs
# ============================================================

overview_tab, run_tab, history_tab, kb_tab = (
    st.tabs(
        [
            tr(lang, "overview"),
            tr(lang, "run_check"),
            tr(lang, "history"),
            tr(lang, "knowledge_base"),
        ]
    )
)

 


# ============================================================
# Overview
# ============================================================

with overview_tab:
    st.header(
        tr(
            lang,
            "reliability_overview",
        )
    )

    history = load_history(
        api_url
    )

    items = history.get(
        "items",
        [],
    )

    evaluated_items = [
        item
        for item in items
        if item.get(
            "overall_health_score"
        )
        is not None
    ]

    total_checks = history.get(
        "total",
        len(items),
    )

    scores = [
        item[
            "overall_health_score"
        ]
        for item in evaluated_items
    ]

    average_score = (
        round(
            sum(scores)
            / len(scores)
        )
        if scores
        else None
    )

    critical_count = sum(
        1
        for item in evaluated_items
        if item.get(
            "evaluation_status"
        )
        == "CRITICAL"
    )

    poor_count = sum(
        1
        for item in evaluated_items
        if item.get(
            "health_status"
        )
        in {
            "POOR",
            "CRITICAL",
        }
    )

    latest_score = (
        evaluated_items[0].get(
            "overall_health_score"
        )
        if evaluated_items
        else None
    )

    cards = st.columns(5)

    cards[0].metric(
        tr(
            lang,
            "total_checks",
        ),
        total_checks,
        border=True,
    )

    cards[1].metric(
        tr(
            lang,
            "evaluated_checks",
        ),
        len(
            evaluated_items
        ),
        border=True,
    )

    cards[2].metric(
        tr(
            lang,
            "average_score",
        ),
        score_value(
            average_score,
            lang,
        ),
        border=True,
    )

    cards[3].metric(
        tr(
            lang,
            "latest_score",
        ),
        score_value(
            latest_score,
            lang,
        ),
        border=True,
    )

    cards[4].metric(
        tr(
            lang,
            "critical_evaluations",
        ),
        critical_count,
        border=True,
    )

    if poor_count:
        st.warning(
            tr(
                lang,
                "poor_warning",
            ).format(
                count=poor_count
            )
        )

    st.subheader(
        tr(
            lang,
            "health_score_history",
        )
    )

    if len(evaluated_items) >= 2:
        trend_rows = []

        for item in reversed(
            evaluated_items
        ):
            trend_rows.append(
                {
                    tr(
                        lang,
                        "created",
                    ): pd.to_datetime(
                        item.get(
                            "created_at"
                        )
                    ),

                    tr(
                        lang,
                        "score",
                    ): item.get(
                        "overall_health_score"
                    ),
                }
            )

        trend_df = pd.DataFrame(
            trend_rows
        )

        st.line_chart(
            trend_df,
            x=tr(
                lang,
                "created",
            ),
            y=tr(
                lang,
                "score",
            ),
        )

    elif len(evaluated_items) == 1:
        st.info(
            tr(
                lang,
                "trend_needs_two",
            )
        )

    else:
        st.info(
            tr(
                lang,
                "no_scores",
            )
        )

    st.subheader(
        tr(
            lang,
            "recent_checks",
        )
    )

    if items:
        table_rows = []

        for item in items[:10]:
            is_legacy = (
                item.get(
                    "overall_health_score"
                )
                is None
            )

            legacy_label = tr(
                lang,
                "legacy_check",
            )

            table_rows.append(
                {
                    tr(
                        lang,
                        "created",
                    ): format_datetime(
                        item.get(
                            "created_at"
                        ),
                        lang,
                    ),

                    tr(
                        lang,
                        "project",
                    ): item.get(
                        "project_id"
                    ),

                    tr(
                        lang,
                        "score",
                    ): (
                        "-"
                        if is_legacy
                        else item.get(
                            "overall_health_score"
                        )
                    ),

                    tr(
                        lang,
                        "health",
                    ): (
                        legacy_label
                        if is_legacy
                        else localize_value(
                            item.get(
                                "health_status"
                            ),
                            lang,
                        )
                    ),

                    tr(
                        lang,
                        "evaluation",
                    ): (
                        legacy_label
                        if is_legacy
                        else localize_value(
                            item.get(
                                "evaluation_status"
                            ),
                            lang,
                        )
                    ),

                    tr(
                        lang,
                        "root_cause",
                    ): localize_value(
                        item.get(
                            "diagnosis_category"
                        ),
                        lang,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                table_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            tr(
                lang,
                "no_checks",
            )
        )


# ============================================================
# Run Health Check
# ============================================================

with run_tab:
    st.header(
        tr(
            lang,
            "run_new",
        )
    )

    st.caption(
        tr(
            lang,
            "run_caption",
        )
    )

    with st.form(
        "health_check_form"
    ):
        project_col, version_col = (
            st.columns(2)
        )

        project_id = (
            project_col.text_input(
                tr(
                    lang,
                    "project_id",
                ),
                value=(
                    "customer-support-rag"
                ),
            )
        )

        application_version = (
            version_col.text_input(
                tr(
                    lang,
                    "application_version",
                ),
                value="1.0.0",
            )
        )

        question = st.text_area(
            tr(
                lang,
                "question",
            ),
            value=(
                "What is the refund period?"
            ),
            height=90,
        )

        answer = st.text_area(
            tr(
                lang,
                "rag_answer",
            ),
            value=(
                "Customers can request "
                "a refund within 30 days."
            ),
            height=100,
        )

        context_text = st.text_area(
            tr(
                lang,
                "retrieved_context",
            ),
            value=(
                "Customers may request "
                "a refund within 14 days."
            ),
            height=140,
        )

        source = st.text_input(
            tr(
                lang,
                "context_source",
            ),
            value=(
                "refund_policy.pdf"
            ),
        )

        reference_answer = (
            st.text_area(
                tr(
                    lang,
                    "reference_answer_optional",
                ),
                value=(
                    "Customers may request "
                    "a refund within 14 days."
                ),
                height=100,
            )
        )

        prompt = st.text_area(
            tr(
                lang,
                "prompt_optional",
            ),
            value=(
                "Answer only using the "
                "provided context."
            ),
            height=80,
        )

        st.subheader(
            tr(
                lang,
                "model_configuration",
            )
        )

        model_col1, model_col2, model_col3 = (
            st.columns(3)
        )

        provider = (
            model_col1.text_input(
                tr(
                    lang,
                    "provider",
                ),
                value="google",
            )
        )

        model_name = (
            model_col2.text_input(
                tr(
                    lang,
                    "model_name",
                ),
                value="gemini-test",
            )
        )

        temperature = (
            model_col3.number_input(
                tr(
                    lang,
                    "temperature",
                ),
                min_value=0.0,
                max_value=2.0,
                value=0.2,
                step=0.1,
            )
        )

        submitted = (
            st.form_submit_button(
                tr(
                    lang,
                    "run_button",
                ),
                use_container_width=True,
                type="primary",
            )
        )

    if submitted:
        if not project_id.strip():
            st.error(
                tr(
                    lang,
                    "project_required",
                )
            )

        elif not question.strip():
            st.error(
                tr(
                    lang,
                    "question_required",
                )
            )

        elif not answer.strip():
            st.error(
                tr(
                    lang,
                    "answer_required",
                )
            )

        else:
            contexts = []

            if context_text.strip():
                contexts.append(
                    {
                        "text": (
                            context_text.strip()
                        ),
                        "source": (
                            source.strip()
                            or None
                        ),
                        "rank": 1,
                        "retrieval_score": None,
                    }
                )

            payload = {
                "project_id": (
                    project_id.strip()
                ),

                "application_version": (
                    application_version.strip()
                    or None
                ),

                "question": (
                    question.strip()
                ),

                "answer": (
                    answer.strip()
                ),

                "contexts": contexts,

                "reference_answer": (
                    reference_answer.strip()
                    or None
                ),

                "prompt": (
                    prompt.strip()
                    or None
                ),

                "model": {
                    "provider": (
                        provider.strip()
                    ),

                    "name": (
                        model_name.strip()
                    ),

                    "temperature": float(
                        temperature
                    ),
                },

                "metadata": {
                    "source": (
                        "streamlit-dashboard"
                    )
                },
            }

            try:
                with st.spinner(
                    tr(
                        lang,
                        "running",
                    )
                ):
                    result = (
                        create_health_check(
                            api_url,
                            payload,
                        )
                    )

                st.success(
                    tr(
                        lang,
                        "completed",
                    )
                )

                created_id = result.get(
                    "health_check_id"
                )

                if created_id:
                    detail = (
                        get_health_check(
                            api_url,
                            created_id,
                        )
                    )

                    render_health_check_detail(
                        detail,
                        lang,
                    )

                else:
                    st.json(
                        result
                    )

            except APIClientError as exc:
                st.error(
                    str(exc)
                )


# ============================================================
# History
# ============================================================

with history_tab:
    st.header(
        tr(
            lang,
            "history_title",
        )
    )

    history = load_history(
        api_url
    )

    items = history.get(
        "items",
        [],
    )

    if not items:
        st.info(
            tr(
                lang,
                "no_checks",
            )
        )

    else:
        option_map = {}

        for item in items:
            health_check_id = (
                item.get(
                    "health_check_id"
                )
            )

            score = (
                item.get(
                    "overall_health_score"
                )
            )

            health_label = (
                localize_value(
                    item.get(
                        "health_status"
                    ),
                    lang,
                )
            )

            label = (
                f"{format_datetime(item.get('created_at'), lang)}"
                f" | {item.get('project_id')}"
                f" | {tr(lang, 'score')}: {score}"
                f" | {health_label}"
            )

            option_map[
                health_check_id
            ] = label

        selected_id = (
            st.selectbox(
                tr(
                    lang,
                    "select_check",
                ),
                options=list(
                    option_map.keys()
                ),
                format_func=lambda value: (
                    option_map[value]
                ),
            )
        )

        if selected_id:
            try:
                detail = (
                    get_health_check(
                        api_url,
                        selected_id,
                    )
                )

                render_health_check_detail(
                    detail,
                    lang,
                )

            except APIClientError as exc:
                st.error(
                    str(exc)
                )


                # ============================================================
# Knowledge Base
# ============================================================

with kb_tab:
    st.header(tr(lang, "knowledge_base"))
    st.caption(tr(lang, "kb_caption"))

    st.subheader(tr(lang, "upload_document"))

    with st.form("kb_upload_form"):
        kb_project_id = st.text_input(
            tr(lang, "project_id"),
            value="customer-support-rag",
            key="kb_upload_project_id",
        )
        uploaded_file = st.file_uploader(
            tr(lang, "choose_pdf"),
            type=["pdf"],
        )
        upload_submitted = st.form_submit_button(
            tr(lang, "upload_index"),
            use_container_width=True,
            type="primary",
        )

    if upload_submitted:
        if not kb_project_id.strip():
            st.error(tr(lang, "project_required"))
        elif uploaded_file is None:
            st.error(tr(lang, "choose_file_required"))
        else:
            try:
                with st.spinner(tr(lang, "uploading_indexing")):
                    result = upload_knowledge_base_document(
                        api_url,
                        project_id=kb_project_id.strip(),
                        file_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                    )

                if result.get("success"):
                    if result.get("duplicate"):
                        st.info(tr(lang, "duplicate_document"))
                    else:
                        st.success(tr(lang, "indexed_successfully"))
                    st.metric(
                        tr(lang, "chunks_indexed"),
                        result.get("chunks_indexed", 0),
                        border=True,
                    )
                else:
                    st.warning(tr(lang, "indexing_failed"))
            except APIClientError as exc:
                st.error(str(exc))

    st.divider()
    st.subheader(tr(lang, "verify_answer"))
    st.caption(tr(lang, "kb_verify_caption"))

    with st.form("kb_verify_form"):
        verify_project_id = st.text_input(
            tr(lang, "project_id"),
            value="customer-support-rag",
            key="kb_verify_project_id",
        )
        verify_question = st.text_area(
            tr(lang, "question"),
            value="What is the refund period?",
            height=90,
        )
        verify_answer_text = st.text_area(
            tr(lang, "verification_answer"),
            value="Customers can request a refund within 30 days.",
            height=100,
        )
        verify_rag_context = st.text_area(
            tr(lang, "rag_context_optional"),
            value="Customers may request a refund within 14 days.",
            height=120,
        )
        verify_submitted = st.form_submit_button(
            tr(lang, "verify"),
            use_container_width=True,
            type="primary",
        )

    if verify_submitted:
        if not verify_project_id.strip():
            st.error(tr(lang, "project_required"))
        elif not verify_question.strip():
            st.error(tr(lang, "question_required"))
        elif not verify_answer_text.strip():
            st.error(tr(lang, "answer_required"))
        else:
            try:
                with st.spinner(tr(lang, "searching_kb")):
                    result = verify_knowledge_base_answer(
                        api_url,
                        project_id=verify_project_id.strip(),
                        question=verify_question.strip(),
                        answer=verify_answer_text.strip(),
                        rag_context=verify_rag_context.strip() or None,
                    )
                render_knowledge_base_verification(result, lang)
            except APIClientError as exc:
                st.error(str(exc))
