import pytest

from benchmarks.baseline.runner import (
    BaseRAGResult,
    run_base_rag,
)


def test_runner_uses_versioned_base_rag_prompt():
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "RAG combines retrieval and generation."

    result = run_base_rag(
        question="What is RAG?",
        contexts=[
            "RAG combines retrieval and generation.",
        ],
        llm=fake_llm,
    )

    assert isinstance(result, BaseRAGResult)
    assert result.prompt_name == "base-rag"
    assert result.prompt_version == "base-rag-v1"
    assert result.prompt == captured["prompt"]
    assert "What is RAG?" in result.prompt
    assert (
        "RAG combines retrieval and generation."
        in result.prompt
    )


def test_runner_preserves_normalized_contexts():
    result = run_base_rag(
        question="What is RAG?",
        contexts=[
            "  First context.  ",
            "",
            " Second context. ",
        ],
        llm=lambda prompt: "Example answer.",
    )

    assert result.contexts == (
        "First context.",
        "Second context.",
    )


def test_runner_strips_llm_answer():
    result = run_base_rag(
        question="What is RAG?",
        contexts=["Example context."],
        llm=lambda prompt: "  Example answer.  ",
    )

    assert result.answer == "Example answer."


def test_runner_rejects_empty_llm_answer():
    with pytest.raises(
        ValueError,
        match="llm returned an empty answer",
    ):
        run_base_rag(
            question="What is RAG?",
            contexts=["Example context."],
            llm=lambda prompt: "   ",
        )
