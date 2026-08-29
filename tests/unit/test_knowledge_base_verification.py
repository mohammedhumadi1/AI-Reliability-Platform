from __future__ import annotations

import knowledge_base.verification_agent as agent


def _match(
    text: str,
    source: str = "policy.pdf",
) -> dict:
    return {
        "text": text,
        "source": source,
        "document_id": "doc-1",
        "distance": 0.2,
    }


def test_verification_not_available_without_documents(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "collection_record_count",
        lambda project_id: 0,
    )

    result = agent.verify_answer(
        project_id="p1",
        question="What is the refund period?",
        answer="14 days",
        rag_contexts=[],
    )

    assert result.status == "NOT_AVAILABLE"
    assert result.is_supported is None
    assert result.health_score_component is None


def test_verification_detects_numeric_contradiction(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "collection_record_count",
        lambda project_id: 1,
    )

    monkeypatch.setattr(
        agent,
        "query_similar_chunks",
        lambda **kwargs: [
            _match(
                "Customers may request a refund within 14 days."
            )
        ],
    )

    def fake_similarity(
        left: str,
        right: str,
    ) -> float:
        if "refund period" in left.lower():
            return 0.9

        return 0.95

    monkeypatch.setattr(
        agent,
        "semantic_similarity",
        fake_similarity,
    )

    result = agent.verify_answer(
        project_id="p1",
        question="What is the refund period?",
        answer=(
            "Customers can request a refund "
            "within 30 days."
        ),
        rag_contexts=[
            "Customers may request a refund within 14 days."
        ],
    )

    assert result.status == "CONTRADICTED"
    assert result.is_supported is False
    assert result.numeric_contradiction is True
    assert result.health_score_component == 0.0


def test_verification_supports_aligned_answer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "collection_record_count",
        lambda project_id: 1,
    )

    monkeypatch.setattr(
        agent,
        "query_similar_chunks",
        lambda **kwargs: [
            _match(
                "Refunds are allowed within fourteen days."
            )
        ],
    )

    monkeypatch.setattr(
        agent,
        "semantic_similarity",
        lambda left, right: 0.9,
    )

    result = agent.verify_answer(
        project_id="p1",
        question="What is the refund period?",
        answer="Refunds are allowed within fourteen days.",
        rag_contexts=[
            "Refunds are allowed within fourteen days."
        ],
    )

    assert result.status == "SUPPORTED"
    assert result.is_supported is True
    assert result.answer_support_score == 0.9
    assert result.context_alignment_score == 0.9


def test_no_relevant_company_evidence_is_kb_gap_signal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "collection_record_count",
        lambda project_id: 1,
    )

    monkeypatch.setattr(
        agent,
        "query_similar_chunks",
        lambda **kwargs: [
            _match("Unrelated company information.")
        ],
    )

    monkeypatch.setattr(
        agent,
        "semantic_similarity",
        lambda left, right: 0.2,
    )

    result = agent.verify_answer(
        project_id="p1",
        question="What is the refund period?",
        answer="14 days",
        rag_contexts=[],
    )

    assert result.status == "NO_RELEVANT_EVIDENCE"
    assert result.evidence_found is False
    assert result.health_score_component == 0.0

def test_numeric_contradiction_uses_question_aligned_evidence_unit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "collection_record_count",
        lambda project_id: 1,
    )

    evidence = (
        "Password reset requests must be completed within 15 minutes. "
        "Temporary passwords must be changed after login. "
        "Critical incidents must receive an initial response within "
        "30 minutes."
    )

    monkeypatch.setattr(
        agent,
        "query_similar_chunks",
        lambda **kwargs: [
            _match(evidence)
        ],
    )

    question = (
        "When must password reset requests be completed?"
    )

    def fake_similarity(
        left: str,
        right: str,
    ) -> float:
        if left == question:
            if (
                "password reset requests"
                in right.lower()
            ):
                return 0.9

            if (
                "critical incidents"
                in right.lower()
            ):
                return 0.2

        return 0.95

    monkeypatch.setattr(
        agent,
        "semantic_similarity",
        fake_similarity,
    )

    result = agent.verify_answer(
        project_id="p1",
        question=question,
        answer=(
            "Password reset requests must be completed "
            "within 30 minutes."
        ),
        rag_contexts=[
            (
                "Password reset requests must be completed "
                "within 15 minutes."
            )
        ],
    )

    assert result.status == "CONTRADICTED"
    assert result.is_supported is False
    assert result.numeric_contradiction is True
    assert result.health_score_component == 0.0
