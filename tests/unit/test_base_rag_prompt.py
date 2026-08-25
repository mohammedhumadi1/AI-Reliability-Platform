import pytest

from benchmarks.baseline.prompt import (
    BASE_RAG_PROMPT_TEMPLATE,
    PROMPT_NAME,
    PROMPT_VERSION,
    build_base_rag_prompt,
)


def test_prompt_metadata_is_versioned():
    assert PROMPT_NAME == "base-rag"
    assert PROMPT_VERSION == "base-rag-v1"


def test_prompt_template_contains_grounding_instruction():
    assert (
        "using only the provided context"
        in BASE_RAG_PROMPT_TEMPLATE
    )
    assert (
        "Do not use external knowledge"
        in BASE_RAG_PROMPT_TEMPLATE
    )


def test_build_prompt_includes_question_and_contexts():
    prompt = build_base_rag_prompt(
        question="What is RAG?",
        contexts=[
            "RAG combines retrieval and generation.",
            "Retrieved documents provide evidence.",
        ],
    )

    assert "What is RAG?" in prompt
    assert (
        "[Context 1]\nRAG combines retrieval and generation."
        in prompt
    )
    assert (
        "[Context 2]\nRetrieved documents provide evidence."
        in prompt
    )


def test_build_prompt_is_deterministic():
    first = build_base_rag_prompt(
        question="What is RAG?",
        contexts=["RAG retrieves supporting context."],
    )

    second = build_base_rag_prompt(
        question="What is RAG?",
        contexts=["RAG retrieves supporting context."],
    )

    assert first == second


def test_build_prompt_handles_empty_contexts():
    prompt = build_base_rag_prompt(
        question="What is RAG?",
        contexts=[],
    )

    assert "[No context provided]" in prompt


def test_build_prompt_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        build_base_rag_prompt(
            question="   ",
            contexts=["Example context."],
        )
