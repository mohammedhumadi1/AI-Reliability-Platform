# Gold Benchmark v1 — Label Definitions (for Reviewer B)

Please assign exactly ONE label per candidate, based only on the
question, context(s), answer, and prompt (if shown) — without seeing
any other metadata.

- **HEALTHY**: The context is relevant and sufficient, and the answer
  is correct, relevant, and supported by the context.

- **RETRIEVAL_FAILURE**: The correct information exists in the source
  knowledge, but the retrieved context(s) missed it or retrieved an
  inappropriate/irrelevant context instead.

- **GENERATION_FAILURE**: The retrieved context contains the correct
  evidence, but the answer contradicts, ignores, or hallucinates beyond
  that evidence.

- **KNOWLEDGE_BASE_FAILURE**: The information needed to answer the
  question does not exist in the available source knowledge at all.

- **PROMPT_FAILURE**: There is real evidence inside the prompt text of
  a conflict/ambiguity in instructions that caused the wrong behavior
  (only applies when a `prompt` field is shown).

If you're unsure between two labels, note it in the `notes` field along
with your reasoning.
