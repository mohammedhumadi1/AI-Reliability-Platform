# RAG Evaluation Benchmark

This directory contains experimental benchmark infrastructure for evaluating and calibrating faithfulness methods.

## Production boundary

The production Faithfulness evaluator remains the multilingual semantic-similarity implementation.

The NLI benchmark is experimental and is not used by the FastAPI production health-check pipeline.

## Dataset

The frozen validation manifest is `benchmarks/manifests/ragbench_validation_v1.json` and contains 500 balanced samples across 10 RAGBench subsets.

Base seed: `20260821`.

## Benchmark dependencies

```powershell
uv sync --frozen --group benchmark
```

## Production embedding baseline

```powershell
uv run python benchmarks/run_embedding_baseline.py
```

## Threshold calibration

```powershell
uv run python benchmarks/tune_embedding_threshold.py
```

Thresholds must be calibrated on validation data only, not the final test split.

## Experimental NLI baseline

```powershell
uv run python benchmarks/run_mdeberta_baseline.py
```

This uses `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` as a pretrained multilingual NLI judge. It is not fine-tuned by this project.

## Tests

```powershell
uv run pytest -q
```

An experimental evaluator should only be promoted after checking holdout performance, uncertainty, latency, resource usage, real-world workloads, and regression risk.
