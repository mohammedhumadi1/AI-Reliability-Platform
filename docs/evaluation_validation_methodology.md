# Evaluation Validation Methodology

## Scope

This document describes the current evaluation implementation and the
validation procedure used to calibrate and test the AI Reliability Platform.

The production evaluator currently uses pretrained multilingual sentence
embeddings and deterministic rules. Experimental NLI evaluators remain outside
the production health-check pipeline.

## Gold-label classes

The validation framework uses five mutually exclusive primary labels:

- `HEALTHY`
- `RETRIEVAL_FAILURE`
- `GENERATION_FAILURE`
- `KNOWLEDGE_BASE_FAILURE`
- `PROMPT_FAILURE`

### HEALTHY

Use when the retrieved evidence is relevant and sufficient, the answer is
grounded in that evidence, and no primary failure mode is present.

### RETRIEVAL_FAILURE

Use when relevant evidence exists or is expected, but the submitted retrieved
contexts are irrelevant, substantially incomplete, or miss the available
evidence.

### GENERATION_FAILURE

Use when useful evidence is available in the retrieved context but the
generated answer is unsupported, contradictory, or otherwise fails to use that
evidence correctly.

### KNOWLEDGE_BASE_FAILURE

Use when the information required to answer the question is not present in the
authoritative indexed company knowledge base.

### PROMPT_FAILURE

Use only when prompt evidence is available and the prompt/template contains a
problem that plausibly explains the failure, such as unclear grounding
instructions, conflicting instructions, or an incompatible requested format.

Low answer relevancy alone is not sufficient evidence to establish a
prompt failure.

## Independent review process

1. Samples are prepared without exposing the final gold label to reviewers.
2. At least two reviewers independently assign one of the five labels.
3. Reviewer agreement is calculated before adjudication.
4. Cohen's kappa and raw observed agreement are reported.
5. If all reviewers agree, the gold label must equal that consensus.
6. If reviewers disagree, an independent adjudicator reviews the evidence and
   assigns the final gold label.
7. Reviewer annotations and adjudication metadata are retained with the
   benchmark record.

Adjudication metadata includes the independent adjudicator identity, the final
adjudicated label, and optional notes. When reviewers disagree, validation
requires the stored gold label to match the adjudicator's recorded decision.

The software can validate this process, but it cannot replace the required
human independent review.

## Development and held-out separation

Gold samples are split deterministically and stratified by failure label.

- Development data may be used for threshold calibration and diagnosis-rule
  tuning.
- Held-out data must not be used for threshold selection or rule tuning.
- Sample IDs must not overlap between the two splits.
- Final reported performance must be calculated on the frozen held-out set.

Benchmark-wide diversity requirements should be checked on the complete
benchmark with `validate_benchmark_coverage` before split-specific reporting.
The split report intentionally summarizes its own coverage without reapplying
whole-benchmark diversity requirements.

## Current semantic model

Production semantic scores use:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Embeddings are normalized. Pairwise similarity is computed using the matrix
dot product of normalized embeddings and clipped to the interval `[0, 1]`.

Let `sim(a, b)` denote this normalized semantic-similarity function.

## Current score definitions

### Correctness

With a reference answer `r`:

`correctness = sim(answer, r)`

If no reference answer is provided, the current implementation uses the
combined retrieved context `c` as a proxy:

`correctness_proxy = sim(answer, c)`

This fallback is explicitly a proxy and should not be interpreted as
reference-grounded correctness.

If neither reference nor context is available, the score is `0.0`.

### Faithfulness

Let `c` be the concatenation of the submitted retrieved contexts:

`faithfulness = sim(answer, c)`

If no retrieved context exists, the score is `0.0`.

This is a semantic grounding proxy, not an NLI entailment score.

### Context Precision

For question `q` and retrieved chunks `c_1 ... c_n`:

`context_precision = mean(sim(q, c_i))`

This is a semantic relevance proxy. It is not classical label-based
information-retrieval precision.

If no contexts are supplied, the score is `0.0`.

### Context Recall

Context recall requires a reference answer.

The reference answer is split into statements `r_1 ... r_m`. Each reference
statement receives the similarity of its best matching retrieved chunk:

`coverage_i = max_j sim(r_i, c_j)`

Then:

`context_recall = mean(coverage_i)`

If no reference answer exists, context recall is unavailable (`None`) rather
than estimated from the submitted context.

If a reference exists but no context is retrieved, the score is `0.0`.

### Answer Relevancy

For question `q` and generated answer `a`:

`answer_relevancy = sim(q, a)`

### Hallucination Risk

The current heuristic is:

`base_risk = 1 - faithfulness`

Without numeric contradiction:

`hallucination_risk = base_risk`

If a numeric contradiction is detected:

`hallucination_risk = max(0.80, base_risk)`

The final value is clipped to `[0, 1]`.

## Numeric contradiction safeguard

Numeric consistency is evaluated separately from semantic similarity.

When a numeric contradiction is detected, production currently caps both
correctness and faithfulness at `0.20`.

This safeguard exists because two statements may have high semantic similarity
while differing in a critical numeric value.

## Current status thresholds

The current production status classifier uses these engineering thresholds:

- `HEALTHY`: correctness >= `0.80` and faithfulness >= `0.70`
- `WARNING`: correctness >= `0.60` and faithfulness >= `0.50`
- otherwise: `CRITICAL`
- any detected numeric contradiction: `CRITICAL`

These are current engineering defaults, not final scientifically validated
thresholds.

Thresholds must be calibrated using development data only and frozen before
held-out evaluation.

## Reporting

The multiclass validation report should include:

- overall accuracy
- per-class precision
- per-class recall
- per-class F1
- macro F1
- weighted F1
- confusion matrix
- sample counts by class
- sample counts by language and domain
- reviewer observed agreement
- Cohen's kappa

Final results should distinguish development/calibration results from held-out
test performance.
