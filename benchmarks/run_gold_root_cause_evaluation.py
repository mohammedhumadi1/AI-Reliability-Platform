from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(
        0,
        str(
            Path(__file__)
            .resolve()
            .parents[1]
        ),
    )


from benchmarks.validation_runner import (
    ValidationPrediction,
    build_validation_report,
    diagnosis_to_label,
)
from benchmarks.validation_schema import (
    Adjudication,
    FailureLabel,
    ReviewerAnnotation,
    SupportedLanguage,
    ValidationSample,
    ValidationSplit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "datasets"
    / "gold_v1"
)

SPLIT_FILES = {
    ValidationSplit.DEVELOPMENT: (
        "samples_development.json"
    ),
    ValidationSplit.HELD_OUT: (
        "samples_held_out.json"
    ),
}

EXPECTED_SPLIT_COUNTS = {
    ValidationSplit.DEVELOPMENT: 35,
    ValidationSplit.HELD_OUT: 15,
}

CALIBRATION_RULES_SHA = (
    "33559c1bf77ef3116a9ca30ee8359f078276bee8"
)

CALIBRATION_SENSITIVE_PATHS = (
    "app/services/health_check_service.py",
    "evaluation",
    "knowledge_base",
    "root_cause",
    "benchmarks/validation_metrics.py",
    "benchmarks/validation_runner.py",
    "benchmarks/validation_schema.py",
)


def _build_sample(
    raw: dict,
) -> ValidationSample:
    reviewers = tuple(
        ReviewerAnnotation(
            reviewer_id=item["reviewer_id"],
            label=FailureLabel(
                item["label"]
            ),
            notes=item.get("notes"),
        )
        for item in raw.get(
            "reviewers",
            [],
        )
    )

    adjudication = None
    raw_adjudication = raw.get(
        "adjudication"
    )

    if raw_adjudication:
        adjudication = Adjudication(
            adjudicator_id=(
                raw_adjudication[
                    "adjudicator_id"
                ]
            ),
            label=FailureLabel(
                raw_adjudication["label"]
            ),
            notes=raw_adjudication.get(
                "notes"
            ),
        )

    return ValidationSample(
        sample_id=raw["sample_id"],
        split=ValidationSplit(
            raw["split"]
        ),
        gold_label=FailureLabel(
            raw["gold_label"]
        ),
        language=SupportedLanguage(
            raw["language"]
        ),
        domain=raw["domain"],
        question=raw["question"],
        answer=raw["answer"],
        contexts=tuple(
            raw["contexts"]
        ),
        model_provider=(
            raw["model_provider"]
        ),
        model_name=raw["model_name"],
        retriever_name=(
            raw["retriever_name"]
        ),
        reference_answer=raw.get(
            "reference_answer"
        ),
        prompt=raw.get("prompt"),
        reviewers=reviewers,
        adjudication=adjudication,
        source_fact_id=raw.get(
            "source_fact_id"
        ),
    )


def load_split_samples(
    split: ValidationSplit,
    confirm_held_out: bool = False,
    dataset_dir: Path = DATASET_DIR,
) -> list[ValidationSample]:
    """
    Load exactly one benchmark split.

    The held-out file is not opened unless
    explicit confirmation was supplied.
    """
    if (
        split
        == ValidationSplit.HELD_OUT
        and not confirm_held_out
    ):
        raise ValueError(
            "Held-out evaluation requires "
            "--confirm-held-out."
        )

    path = (
        dataset_dir
        / SPLIT_FILES[split]
    )

    raw_samples = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    samples = [
        _build_sample(raw)
        for raw in raw_samples
    ]

    expected_count = (
        EXPECTED_SPLIT_COUNTS[
            split
        ]
    )

    if len(samples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} "
            f"{split.value} samples, "
            f"found {len(samples)}."
        )

    if any(
        sample.split != split
        for sample in samples
    ):
        raise ValueError(
            "Dataset file contains samples "
            "from the wrong split."
        )

    return samples


def get_git_state() -> dict:
    sha_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    status_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    sha = (
        sha_result.stdout.strip()
        if sha_result.returncode == 0
        else "unknown"
    )

    clean = (
        status_result.returncode == 0
        and not status_result.stdout.strip()
    )

    return {
        "commit_sha": sha,
        "working_tree_clean": clean,
    }


def calibration_code_is_frozen() -> bool:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            CALIBRATION_RULES_SHA,
            "--",
            *CALIBRATION_SENSITIVE_PATHS,
        ],
        cwd=REPO_ROOT,
        check=False,
    )

    if result.returncode not in {0, 1}:
        raise RuntimeError(
            "Unable to verify frozen calibration code."
        )

    return result.returncode == 0


def build_breakdown(
    samples: list[ValidationSample],
    predictions: list[
        ValidationPrediction
    ],
    attribute: str,
) -> dict:
    prediction_map = {
        item.sample_id: (
            item.predicted_label
        )
        for item in predictions
    }

    groups: dict[str, list[
        ValidationSample
    ]] = {}

    for sample in samples:
        value = getattr(
            sample,
            attribute,
        )

        if hasattr(value, "value"):
            key = str(value.value)
        else:
            key = str(value)

        groups.setdefault(
            key,
            [],
        ).append(sample)

    result = {}

    for key, members in sorted(
        groups.items()
    ):
        correct = sum(
            prediction_map[
                sample.sample_id
            ]
            == sample.gold_label
            for sample in members
        )

        gold_counts = Counter(
            sample.gold_label.value
            for sample in members
        )

        predicted_counts = Counter(
            prediction_map[
                sample.sample_id
            ].value
            for sample in members
        )

        result[key] = {
            "sample_count": len(
                members
            ),
            "correct": correct,
            "mistakes": (
                len(members) - correct
            ),
            "accuracy": (
                correct / len(members)
            ),
            "gold_label_counts": dict(
                sorted(
                    gold_counts.items()
                )
            ),
            "predicted_label_counts": (
                dict(
                    sorted(
                        predicted_counts.items()
                    )
                )
            ),
        }

    return result


def evaluate_split(
    samples: list[ValidationSample],
    split: ValidationSplit,
) -> dict:
    import chromadb

    import knowledge_base.vector_store as vector_store

    client = chromadb.EphemeralClient()

    try:
        vector_store.get_client.cache_clear()
    except AttributeError:
        pass

    vector_store.get_client = (
        lambda: client
    )

    from app.services.health_check_service import (
        build_root_cause_metrics,
    )
    from evaluation.pipeline import (
        run_evaluation,
    )
    from knowledge_base.vector_store import (
        add_chunks,
    )
    from knowledge_base.verification_agent import (
        verify_answer,
    )
    from root_cause.rules.pipeline import (
        run_rules_pipeline,
    )

    source_facts = json.loads(
        (
            DATASET_DIR
            / "source_facts.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    project_id = (
        "gold-v1-"
        f"{split.value}-"
        "root-cause-evaluation"
    )

    for domain, languages in (
        source_facts.items()
    ):
        for language, facts in (
            languages.items()
        ):
            add_chunks(
                project_id=project_id,
                chunks=facts,
                source=(
                    f"{domain}/{language}"
                ),
                document_id=(
                    f"{domain}_{language}"
                ),
            )

    predictions: list[
        ValidationPrediction
    ] = []

    mistakes = []

    for sample in samples:
        evaluation = run_evaluation(
            question=sample.question,
            answer=sample.answer,
            contexts=list(
                sample.contexts
            ),
            reference_answer=(
                sample.reference_answer
            ),
        )

        verification = verify_answer(
            project_id=project_id,
            question=sample.question,
            answer=sample.answer,
            rag_contexts=list(
                sample.contexts
            ),
        )

        metrics = (
            build_root_cause_metrics(
                result=evaluation,
                verification=(
                    verification
                ),
                prompt=sample.prompt,
            )
        )

        diagnosis = run_rules_pipeline(
            metrics
        )

        predicted_label = (
            diagnosis_to_label(
                diagnosis
            )
        )

        predictions.append(
            ValidationPrediction(
                sample_id=(
                    sample.sample_id
                ),
                predicted_label=(
                    predicted_label
                ),
            )
        )

        if (
            predicted_label
            != sample.gold_label
        ):
            mistakes.append(
                {
                    "sample_id": (
                        sample.sample_id
                    ),
                    "gold_label": (
                        sample.gold_label.value
                    ),
                    "predicted_label": (
                        predicted_label.value
                    ),
                    "language": (
                        sample.language.value
                    ),
                    "domain": (
                        sample.domain
                    ),
                }
            )

    report = build_validation_report(
        samples=samples,
        predictions=predictions,
        split=split,
    )

    payload = report.to_dict()

    prediction_map = {
        prediction.sample_id: (
            prediction.predicted_label
        )
        for prediction in predictions
    }

    prediction_records = [
        {
            "sample_id": sample.sample_id,
            "gold_label": (
                sample.gold_label.value
            ),
            "predicted_label": (
                prediction_map[
                    sample.sample_id
                ].value
            ),
            "language": (
                sample.language.value
            ),
            "domain": sample.domain,
        }
        for sample in samples
    ]

    payload.update(
        {
            "schema_version": (
                "gold_root_cause_"
                "evaluation_v1"
            ),
            "calibration_rules_sha": (
                CALIBRATION_RULES_SHA
            ),
            "git": get_git_state(),
            "predictions": (
                prediction_records
            ),
            "breakdown_by_language": (
                build_breakdown(
                    samples,
                    predictions,
                    "language",
                )
            ),
            "breakdown_by_domain": (
                build_breakdown(
                    samples,
                    predictions,
                    "domain",
                )
            ),
            "mistakes": mistakes,
        }
    )

    return payload


def print_report(
    payload: dict,
) -> None:
    print(
        "=== GOLD ROOT-CAUSE "
        "EVALUATION ==="
    )
    print(
        "Split:",
        payload["split"],
    )
    print(
        "Commit:",
        payload["git"][
            "commit_sha"
        ],
    )
    print(
        "Working tree clean:",
        payload["git"][
            "working_tree_clean"
        ],
    )
    print(
        "Samples:",
        payload["sample_count"],
    )
    print(
        "Accuracy:",
        f"{payload['accuracy']:.4f}",
    )
    print(
        "Macro F1:",
        f"{payload['macro_f1']:.4f}",
    )
    print(
        "Weighted F1:",
        f"{payload['weighted_f1']:.4f}",
    )

    print("\nPer-class:")
    for label in payload["labels"]:
        values = (
            payload["per_class"][
                label
            ]
        )

        print(
            f"  {label}: "
            f"P={values['precision']:.4f} "
            f"R={values['recall']:.4f} "
            f"F1={values['f1']:.4f} "
            f"N={values['support']}"
        )

    print("\nConfusion matrix:")
    print(
        "  labels:",
        payload["labels"],
    )

    for row in payload[
        "confusion_matrix"
    ]:
        print(
            " ",
            row,
        )

    print("\nLanguage breakdown:")
    for language, values in (
        payload[
            "breakdown_by_language"
        ].items()
    ):
        print(
            f"  {language}: "
            f"N={values['sample_count']} "
            f"accuracy="
            f"{values['accuracy']:.4f} "
            f"mistakes="
            f"{values['mistakes']}"
        )

    print("\nDomain breakdown:")
    for domain, values in (
        payload[
            "breakdown_by_domain"
        ].items()
    ):
        print(
            f"  {domain}: "
            f"N={values['sample_count']} "
            f"accuracy="
            f"{values['accuracy']:.4f} "
            f"mistakes="
            f"{values['mistakes']}"
        )

    print(
        "\nMistakes:",
        len(payload["mistakes"]),
    )

    for mistake in payload[
        "mistakes"
    ]:
        print(
            " ",
            mistake["sample_id"],
            "|",
            mistake["gold_label"],
            "->",
            mistake[
                "predicted_label"
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Gold Benchmark "
            "root-cause evaluation."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            split.value
            for split
            in ValidationSplit
        ],
        default=(
            ValidationSplit
            .DEVELOPMENT
            .value
        ),
    )

    parser.add_argument(
        "--confirm-held-out",
        action="store_true",
        help=(
            "Required before the held-out "
            "file can be read."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split = ValidationSplit(
        args.split
    )

    git_state = get_git_state()

    if split == ValidationSplit.HELD_OUT:
        if not git_state[
            "working_tree_clean"
        ]:
            raise SystemExit(
                "Held-out evaluation requires "
                "a clean working tree."
            )

        if not args.confirm_held_out:
            raise SystemExit(
                "Held-out evaluation requires "
                "--confirm-held-out."
            )

        if not calibration_code_is_frozen():
            raise SystemExit(
                "Held-out evaluation refused: "
                "calibration-sensitive code differs "
                "from frozen commit "
                f"{CALIBRATION_RULES_SHA}."
            )

        if args.output is None:
            raise SystemExit(
                "Held-out evaluation requires "
                "--output to preserve the final report."
            )

        if args.output.exists():
            raise SystemExit(
                "Held-out output already exists; "
                "refusing to overwrite it."
            )

    try:
        samples = load_split_samples(
            split=split,
            confirm_held_out=(
                args.confirm_held_out
            ),
        )
    except ValueError as exc:
        raise SystemExit(
            str(exc)
        ) from exc

    payload = evaluate_split(
        samples=samples,
        split=split,
    )

    print_report(
        payload
    )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "\nSaved report:",
            args.output,
        )


if __name__ == "__main__":
    main()
