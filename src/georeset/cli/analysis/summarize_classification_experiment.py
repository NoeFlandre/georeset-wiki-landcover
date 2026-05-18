"""Summarize classification experiment metrics into compact overview tables."""

import argparse
import csv
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, cast

from georeset.classification.text_sources import shuffled_text_source_pairs, text_source_sort_key
from georeset.experiment_paths import experiment_artifact_dir
from georeset.utils.json_io import read_json_file, write_markdown_table_atomic, write_text_atomic

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "run",
    "task",
    "text_source",
    "n_eligible",
    "n_predicted_ok",
    "n_parse_error",
    "coverage",
    "primary_metric",
    "primary_score",
    "majority_baseline_score",
    "delta_vs_majority",
    "majority_target_share",
    "majority_macro_recall_baseline",
    "delta_macro_recall_vs_majority",
    "accuracy",
    "exact_match_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "n_labels_evaluated",
]


SHUFFLED_DELTA_FIELDNAMES = [
    "task",
    "text_source",
    "shuffled_text_source",
    "primary_metric",
    "aligned_score",
    "shuffled_score",
    "delta",
    "n_aligned",
    "n_shuffled",
    "aligned_macro_f1",
    "shuffled_macro_f1",
    "delta_macro_f1",
    "aligned_accuracy",
    "shuffled_accuracy",
    "delta_accuracy",
    "aligned_exact_match_accuracy",
    "shuffled_exact_match_accuracy",
    "delta_exact_match_accuracy",
    "aligned_micro_f1",
    "shuffled_micro_f1",
    "delta_micro_f1",
]


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], read_json_file(path))


def _run_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    source_order, source_name = text_source_sort_key(str(row["text_source"]))
    return (
        str(row["task"]),
        source_order,
        source_name,
    )


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get(key, "")


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, Any]:
    if metrics.get("task") == "osm":
        return "exact_match_accuracy", _metric_value(metrics, "exact_match_accuracy")
    return "macro_recall", _metric_value(metrics, "macro_recall")


def _target_key(target: Any) -> str:
    if isinstance(target, list):
        return json.dumps(sorted(target), ensure_ascii=False)
    return str(target)


def _evaluated_targets(records: dict[str, Any]) -> list[Any]:
    return [
        record["target"]
        for record in records.values()
        if isinstance(record, dict) and record.get("parse_status") == "ok" and "target" in record
    ]


def majority_target_share(records: dict[str, Any]) -> float | str:
    targets = _evaluated_targets(records)
    if not targets:
        return ""
    majority_key = Counter(_target_key(target) for target in targets).most_common(1)[0][0]
    correct = sum(1 for target in targets if _target_key(target) == majority_key)
    return correct / len(targets)


def majority_macro_recall_baseline(
    records: dict[str, Any], labels_evaluated: list[str] | None = None
) -> float | str:
    labels = labels_evaluated or sorted(
        {str(target) for target in _evaluated_targets(records) if not isinstance(target, list)}
    )
    if not labels:
        return ""
    return 1 / len(labels)


def majority_baseline_score(
    records: dict[str, Any],
    primary_metric: str,
    labels_evaluated: list[str] | None = None,
) -> float | str:
    if primary_metric in {"accuracy", "exact_match_accuracy"}:
        return majority_target_share(records)
    if primary_metric == "macro_recall":
        return majority_macro_recall_baseline(records, labels_evaluated)
    return ""


def _predictions_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(metrics_path.name.replace("_metrics.json", "_predictions.json"))


def collect_metric_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(experiment_dir.glob("*_metrics.json")):
        metrics = _load_json(path)
        primary_name, primary_score = _primary_metric(metrics)
        predictions_path = _predictions_path(path)
        majority_score: float | str = ""
        majority_share: float | str = ""
        majority_macro_recall: float | str = ""
        labels_evaluated = metrics.get("labels_evaluated", [])
        if predictions_path.exists():
            records = _load_json(predictions_path)
            majority_score = majority_baseline_score(records, primary_name, labels_evaluated)
            majority_share = majority_target_share(records)
            if metrics.get("task") == "corine_level2":
                majority_macro_recall = majority_macro_recall_baseline(records, labels_evaluated)
        row = {
            "run": f"{metrics['task']}/{metrics['text_source']}",
            "task": metrics["task"],
            "text_source": metrics["text_source"],
            "n_eligible": metrics["n_eligible"],
            "n_predicted_ok": metrics["n_predicted_ok"],
            "n_parse_error": metrics["n_parse_error"],
            "coverage": metrics["coverage"],
            "primary_metric": primary_name,
            "primary_score": primary_score,
            "majority_baseline_score": majority_score,
            "delta_vs_majority": (
                primary_score - majority_score
                if isinstance(primary_score, (int, float))
                and isinstance(majority_score, (int, float))
                else ""
            ),
            "majority_target_share": majority_share,
            "majority_macro_recall_baseline": majority_macro_recall,
            "delta_macro_recall_vs_majority": (
                metrics["macro_recall"] - majority_macro_recall
                if isinstance(metrics.get("macro_recall"), (int, float))
                and isinstance(majority_macro_recall, (int, float))
                else ""
            ),
            "accuracy": _metric_value(metrics, "accuracy"),
            "exact_match_accuracy": _metric_value(metrics, "exact_match_accuracy"),
            "macro_precision": _metric_value(metrics, "macro_precision"),
            "macro_recall": _metric_value(metrics, "macro_recall"),
            "macro_f1": _metric_value(metrics, "macro_f1"),
            "micro_precision": _metric_value(metrics, "micro_precision"),
            "micro_recall": _metric_value(metrics, "micro_recall"),
            "micro_f1": _metric_value(metrics, "micro_f1"),
            "n_labels_evaluated": len(labels_evaluated),
        }
        rows.append(row)
    return sorted(rows, key=_run_sort_key)


def write_overview_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(output_path, output.getvalue())


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def write_overview_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns = [
        "run",
        "n eligible",
        "n predicted ok",
        "parse errors",
        "coverage",
        "primary score",
        "majority baseline",
        "delta vs majority",
        "majority target share",
        "majority macro recall baseline",
        "delta macro recall vs majority",
        "accuracy",
        "exact match accuracy",
        "macro precision",
        "macro recall",
        "macro F1",
        "micro precision",
        "micro recall",
        "micro F1",
        "labels",
    ]
    markdown_rows = [
        {
            "run": _format_cell(row["run"]),
            "n eligible": _format_cell(row["n_eligible"]),
            "n predicted ok": _format_cell(row["n_predicted_ok"]),
            "parse errors": _format_cell(row["n_parse_error"]),
            "coverage": _format_cell(row["coverage"]),
            "primary score": _format_cell(row["primary_score"]),
            "majority baseline": _format_cell(row["majority_baseline_score"]),
            "delta vs majority": _format_cell(row["delta_vs_majority"]),
            "majority target share": _format_cell(row["majority_target_share"]),
            "majority macro recall baseline": _format_cell(row["majority_macro_recall_baseline"]),
            "delta macro recall vs majority": _format_cell(row["delta_macro_recall_vs_majority"]),
            "accuracy": _format_cell(row["accuracy"]),
            "exact match accuracy": _format_cell(row["exact_match_accuracy"]),
            "macro precision": _format_cell(row["macro_precision"]),
            "macro recall": _format_cell(row["macro_recall"]),
            "macro F1": _format_cell(row["macro_f1"]),
            "micro precision": _format_cell(row["micro_precision"]),
            "micro recall": _format_cell(row["micro_recall"]),
            "micro F1": _format_cell(row["micro_f1"]),
            "labels": _format_cell(row["n_labels_evaluated"]),
        }
        for row in rows
    ]
    write_markdown_table_atomic(
        output_path,
        title="Classification Experiment Overview",
        rows=markdown_rows,
        columns=columns,
    )


def _delta(left: Any, right: Any) -> float | str:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return ""


def shuffled_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["task"], row["text_source"]): row for row in rows}
    deltas: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in rows}):
        text_sources = {str(row["text_source"]) for row in rows if row["task"] == task}
        for source, shuffled_source in shuffled_text_source_pairs(text_sources).items():
            aligned = by_key.get((task, source))
            shuffled = by_key.get((task, shuffled_source))
            if not aligned or not shuffled:
                continue
            deltas.append(
                {
                    "task": task,
                    "text_source": source,
                    "shuffled_text_source": shuffled_source,
                    "primary_metric": aligned["primary_metric"],
                    "aligned_score": aligned["primary_score"],
                    "shuffled_score": shuffled["primary_score"],
                    "delta": _delta(aligned["primary_score"], shuffled["primary_score"]),
                    "n_aligned": aligned["n_eligible"],
                    "n_shuffled": shuffled["n_eligible"],
                    "aligned_macro_f1": aligned["macro_f1"],
                    "shuffled_macro_f1": shuffled["macro_f1"],
                    "delta_macro_f1": _delta(aligned["macro_f1"], shuffled["macro_f1"]),
                    "aligned_accuracy": aligned["accuracy"],
                    "shuffled_accuracy": shuffled["accuracy"],
                    "delta_accuracy": _delta(aligned["accuracy"], shuffled["accuracy"]),
                    "aligned_exact_match_accuracy": aligned["exact_match_accuracy"],
                    "shuffled_exact_match_accuracy": shuffled["exact_match_accuracy"],
                    "delta_exact_match_accuracy": _delta(
                        aligned["exact_match_accuracy"], shuffled["exact_match_accuracy"]
                    ),
                    "aligned_micro_f1": aligned["micro_f1"],
                    "shuffled_micro_f1": shuffled["micro_f1"],
                    "delta_micro_f1": _delta(aligned["micro_f1"], shuffled["micro_f1"]),
                }
            )
    return deltas


def write_shuffled_delta_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=SHUFFLED_DELTA_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(output_path, output.getvalue())


def write_shuffled_delta_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    write_markdown_table_atomic(
        output_path,
        title="Shuffled Control Deltas",
        rows=[{key: _format_cell(row.get(key, "")) for key in SHUFFLED_DELTA_FIELDNAMES} for row in rows],
        columns=SHUFFLED_DELTA_FIELDNAMES,
    )


def write_readme(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str = "Article-Text Classification E2E v1",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = sum(int(row["n_eligible"]) for row in rows)
    parse_errors = sum(int(row["n_parse_error"]) for row in rows)
    tasks = sorted({str(row["task"]) for row in rows})
    text_sources = sorted(
        {str(row["text_source"]) for row in rows},
        key=text_source_sort_key,
    )
    has_shuffled_controls = any(source.endswith("_shuffled") for source in text_sources)
    best_delta = max(
        row["delta_vs_majority"]
        for row in rows
        if isinstance(row["delta_vs_majority"], (int, float))
    )
    n_beating_majority = sum(
        1
        for row in rows
        if isinstance(row["delta_vs_majority"], (int, float)) and row["delta_vs_majority"] > 0
    )
    lines = [
        f"# {title}",
        "",
        "This folder contains the frozen outputs for the article-text classification experiment.",
        f"It covers {len(tasks)} task(s) x {len(text_sources)} text source(s):",
        "",
        "- CORINE level-2 single-label classification",
        "- OSM multi-label classification",
        f"- tasks: {_format_code_list(tasks)}",
        f"- text sources: {_format_code_list(text_sources)}",
        "- `summary`: generated with `summary_mode=place`",
        "- `summary_no_place`: generated with `summary_mode=no_place`",
        "- `content`: raw Wikipedia article content",
        "- `landuse_evidence_summary`: extracted no-place land-use evidence summaries",
        "- `landuse_evidence_summary_shuffled`: shuffled control using land-use evidence summaries",
    ]
    if has_shuffled_controls:
        lines.append(
            "- Includes deterministic shuffled controls. These preserve targets and eligible articles while reassigning texts across articles."
        )
    lines.extend(
        [
            "",
            "## Contents",
            "",
            "- `*_predictions.json`: per-article predictions, raw model responses, parse status, and metadata",
            "- `*_metrics.json`: aggregate metrics for each task/text-source run",
            "- `overview.csv`: machine-readable summary table",
            "- `overview.md`: Markdown summary table for quick review",
            "",
            "## Batch Summary",
            "",
            f"- runs: {len(rows)}",
            f"- eligible article-run records: {total_records}",
            f"- parse errors: {parse_errors}",
            f"- best delta vs majority baseline: {_format_cell(best_delta)}",
            "",
            "## Runs",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['run']}`: n={row['n_eligible']}, "
            f"{row['primary_metric']}={_format_cell(row['primary_score'])}, "
            f"macro_f1={_format_cell(row['macro_f1'])}"
        )
    lines.extend(
        [
            "",
            "## Metric Explanations",
            "",
            "- `n_eligible`: number of articles that have ground truth for this task and are included in the evaluation denominator.",
            "- `n_predicted_ok`: number of eligible articles with a valid parsed prediction.",
            "- `n_parse_error`: number of eligible articles without a valid parsed prediction. These are excluded from accuracy/F1, but reduce coverage.",
            "- `coverage`: `n_predicted_ok / n_eligible`. This answers: among evaluable articles, how many received a usable model prediction?",
            "- `majority_baseline_score`: score from a simple classifier that always predicts the most frequent ground-truth class or label-set in that run, using the run's primary metric.",
            "- `delta_vs_majority`: primary model score minus `majority_baseline_score`. Positive means the model beats the class-imbalance baseline; negative means it underperforms that baseline.",
            "- `majority_target_share`: raw frequency of the most common target class or label-set. For CORINE this is the raw accuracy achieved by always predicting the most common land-cover class.",
            "- `majority_macro_recall_baseline`: CORINE-only balanced majority baseline. If there are 9 evaluated classes, always predicting the majority class has recall 1.0 for that class and 0.0 for the other 8, so macro recall is `1 / number of evaluated classes` = 0.1111.",
            "- `delta_macro_recall_vs_majority`: CORINE `macro_recall` minus `majority_macro_recall_baseline`.",
            "- `accuracy`: CORINE single-label score. A prediction is correct only when the one predicted CORINE level-2 label exactly equals the one true label. In imbalanced land-cover data, accuracy can be misleading because the most common class alone can dominate it.",
            "- `exact_match_accuracy`: OSM multi-label score. A prediction is correct only when the full predicted label set exactly equals the full true label set.",
            "- `precision`: among labels the model predicted, the fraction that were actually correct. High precision means fewer false positive labels.",
            "- `recall`: among labels that were truly present, the fraction the model recovered. High recall means fewer missed labels.",
            "- `F1`: harmonic mean of precision and recall. It is high only when both precision and recall are high.",
            "- `macro_precision`, `macro_recall`, `macro_f1`: compute the metric independently for each label, then average labels equally. Example: with labels like `meadow`, `wood`, and `water`, macro precision treats `meadow` with 100 occurrences and `water` with 5 occurrences equally.",
            "- `micro_precision`, `micro_recall`, `micro_f1`: aggregate all true positives, false positives, and false negatives across all labels first, then compute the metric. Example: if `meadow` appears 100 times and `water` appears 5 times, `meadow` contributes 100x more than `water`.",
            "- `n_labels_evaluated`: number of distinct labels appearing in the evaluated true or predicted labels for that run.",
            "",
            "Interpretation notes:",
            "",
            "- CORINE uses `macro_recall` as the primary score. For single-label multiclass classification, macro recall is the balanced accuracy view: compute recall for each class, then average classes equally.",
            "- OSM uses `exact_match_accuracy` as the strict primary score because it is multi-label.",
            "- Read CORINE `accuracy` together with `majority_target_share`. A model can lose on raw accuracy while still beating the majority baseline on balanced accuracy if it recovers minority classes.",
            "- Macro scores are better for seeing whether rare labels are handled well.",
            "- Micro scores are better for seeing overall label-decision performance weighted by common labels.",
            "- Coverage should be read before accuracy/F1. A high score with low coverage can be misleading because many failures were excluded from scoring.",
            "- The majority baseline should be read next. If `delta_vs_majority` is near or below zero, the model is not clearly adding useful signal beyond the relevant class-imbalance baseline.",
            "",
            "## Majority Baseline Finding",
            "",
            f"- Runs beating the majority baseline: {n_beating_majority}/{len(rows)}",
            f"- Best observed delta vs majority baseline: {_format_cell(best_delta)}",
            "- For CORINE, compare `macro_recall` against `majority_macro_recall_baseline`, not only raw accuracy against `majority_target_share`.",
            "- For OSM, the majority baseline remains strict exact-match accuracy for the most common true label-set.",
        ]
    )
    write_text_atomic(output_path, "\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize classification experiment metrics into overview tables."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=experiment_artifact_dir("article_text_classification_e2e_v1"),
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--readme-output", type=Path, default=None)
    parser.add_argument("--title", default="Article-Text Classification E2E v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = collect_metric_rows(args.experiment_dir)
    csv_output = args.csv_output or args.experiment_dir / "overview.csv"
    markdown_output = args.markdown_output or args.experiment_dir / "overview.md"
    readme_output = args.readme_output or args.experiment_dir / "README.md"
    write_overview_csv(rows, csv_output)
    write_overview_markdown(rows, markdown_output)
    deltas = shuffled_delta_rows(rows)
    write_shuffled_delta_csv(deltas, args.experiment_dir / "shuffled_delta.csv")
    write_shuffled_delta_markdown(deltas, args.experiment_dir / "shuffled_delta.md")
    write_readme(rows, readme_output, title=args.title)
    logger.info(
        "Wrote %s rows to %s, %s, and %s", len(rows), csv_output, markdown_output, readme_output
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
