from __future__ import annotations


PROMPT_NAME = "base-rag"
PROMPT_VERSION = "base-rag-v1"

BASE_RAG_PROMPT_TEMPLATE = """You are a question-answering assistant.

Answer the question using only the provided context.
Do not use external knowledge.
If the answer cannot be determined from the context,
state that the information is unavailable.

Context:
{contexts}

Question:
{question}

Answer:"""


def build_base_rag_prompt(
    question: str,
    contexts: list[str] | tuple[str, ...],
) -> str:
    """Build the deterministic prompt used by the Base-RAG benchmark."""
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("question must not be empty")

    normalized_contexts = [
        context.strip()
        for context in contexts
        if context.strip()
    ]

    context_text = "\n\n".join(
        f"[Context {index}]\n{context}"
        for index, context in enumerate(
            normalized_contexts,
            start=1,
        )
    )

    if not context_text:
        context_text = "[No context provided]"

    return BASE_RAG_PROMPT_TEMPLATE.format(
        contexts=context_text,
        question=normalized_question,
    )
