"""Run Experiment 014 weighted image linear probes from cached embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from georeset.analysis.supported_metrics import per_label_counts, single_label_metrics_supported
from georeset.cli.csv_args import parse_csv_strings
from georeset.cli.image_probe_args import (
    embedding_cache_paths,
    image_probe_splits_path,
    sample_weights_path,
)
from georeset.experiment_paths import experiment_artifact_dir
from georeset.utils.json_io import (
    markdown_table,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from georeset.vision.clip_embedding_cache import load_embedding_cache, stack_embeddings_for_rows
from georeset.vision.linear_probe import predict_linear_probe
from georeset.vision.weighted_linear_probe import fit_weighted_linear_probe

EXPERIMENT_ID = "quality_weighted_multiscale_image_probe_v1"
POLICIES = {
    "all_unweighted": ("all", "constant"),
    "spatial_only_unweighted": ("spatial_only", "constant"),
    "quality_spatial_hard": ("quality_spatial", "constant"),
    "agreement_hard": ("text_spatial_agreement", "constant"),
    "all_quality_weighted": ("all", "weight_raw"),
    "all_quality_weighted_class_balanced": ("all", "weight_class_balanced"),
    "spatial_soft_weighted": ("all", "spatial_component"),
    "text_agreement_soft_weighted": ("all", "text_agreement_soft"),
}
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2)
WEIGHT_COLUMNS = (
    "relevance_component",
    "uncertainty_component",
    "spatial_component",
    "agreement_component",
    "weight_raw",
    "weight_class_balanced",
)


def _normalized(values: pd.Series) -> NDArray[np.float64]:
    array = values.astype(float).to_numpy()
    mean = float(array.mean()) if len(array) else 1.0
    if mean <= 0.0:
        return np.ones(len(array), dtype=np.float64)
    return np.asarray((array / mean).astype(np.float64), dtype=np.float64)


def _policy_weights(rows: pd.DataFrame, policy_weight: str) -> NDArray[np.float64]:
    if policy_weight == "constant":
        return np.ones(len(rows), dtype=np.float64)
    if policy_weight == "text_agreement_soft":
        return _normalized(
            rows["agreement_component"] * rows["spatial_component"] * rows["relevance_component"]
        )
    return _normalized(rows[policy_weight])


def _attach_weights(splits: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if all(column in splits.columns for column in WEIGHT_COLUMNS):
        return splits
    existing_weight_columns = [column for column in WEIGHT_COLUMNS if column in splits.columns]
    split_rows = splits.drop(columns=existing_weight_columns)
    weight_rows = weights.drop(columns=["label"], errors="ignore")
    return split_rows.merge(weight_rows, on="pageid", how="left")


def _bootstrap_interval(
    y_true: NDArray[np.str_],
    y_pred: NDArray[np.str_],
    labels: list[str],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float]:
    if len(y_true) == 0 or n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    balanced = []
    macro = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = single_label_metrics_supported(y_true[indices], y_pred[indices], labels)
        balanced.append(metrics["balanced_accuracy_supported"])
        macro.append(metrics["macro_f1_supported"])
    return {
        "balanced_accuracy_supported_ci_low": float(np.quantile(balanced, 0.025)),
        "balanced_accuracy_supported_ci_high": float(np.quantile(balanced, 0.975)),
        "macro_f1_supported_ci_low": float(np.quantile(macro, 0.025)),
        "macro_f1_supported_ci_high": float(np.quantile(macro, 0.975)),
    }


def _confusion_matrix(
    y_true: NDArray[np.str_], y_pred: NDArray[np.str_], labels: list[str]
) -> list[list[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred, strict=True):
        if str(true) in label_to_index and str(pred) in label_to_index:
            matrix[label_to_index[str(true)], label_to_index[str(pred)]] += 1
    return matrix.tolist()


def run_probe(
    *,
    splits_path: Path,
    weights_path: Path,
    embeddings_paths: list[Path],
    output_dir: Path,
    seed: int,
    epochs: int,
    learning_rate: float,
    n_bootstrap: int,
) -> None:
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    weights = pd.read_csv(weights_path, dtype={"pageid": str, "label": str})
    rows = _attach_weights(splits, weights)
    labels = sorted(rows["label"].dropna().astype(str).unique().tolist())
    eval_splits = sorted(
        split
        for split in rows["split"].unique().tolist()
        if split == "eval_strict"
        or str(split).startswith("repeated_eval_seed_")
        or str(split).startswith("spatial_block_fold_")
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    for embeddings_path in embeddings_paths:
        if not embeddings_path.exists():
            continue
        data = np.load(embeddings_path)
        encoder = str(data["encoder_name"]) if "encoder_name" in data else embeddings_path.stem
        window_m = int(data["window_m"]) if "window_m" in data else -1
        embeddings = load_embedding_cache(embeddings_path)
        for policy, (tier, weight_column) in POLICIES.items():
            base_train_rows = rows[(rows["split"] == "train") & (rows["tier"] == tier)].copy()
            if base_train_rows.empty or base_train_rows["label"].nunique() < 2:
                continue
            for split in eval_splits:
                eval_rows = rows[rows["split"] == split].copy()
                if eval_rows.empty:
                    continue
                train_for_split = base_train_rows
                if str(split).startswith("spatial_block_fold_"):
                    train_for_split = base_train_rows[
                        ~base_train_rows["pageid"].isin(eval_rows["pageid"])
                    ].copy()
                if train_for_split.empty or train_for_split["label"].nunique() < 2:
                    continue
                train_rows, train_x = stack_embeddings_for_rows(
                    train_for_split, embeddings, context=f"{policy}/{split}"
                )
                sample_weight = _policy_weights(train_rows, weight_column)
                eval_rows, eval_x = stack_embeddings_for_rows(eval_rows, embeddings, context=split)
                y_true = eval_rows["label"].to_numpy(dtype=str)
                for l2 in L2_GRID:
                    model = fit_weighted_linear_probe(
                        train_x,
                        train_rows["label"].to_numpy(dtype=str),
                        sample_weight,
                        seed=seed,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        l2=l2,
                    )
                    y_pred = predict_linear_probe(model, eval_x)
                    metrics = single_label_metrics_supported(y_true, y_pred, labels)
                    row_id = f"{encoder}|{window_m}|{policy}|{split}|{l2:g}"
                    metric_rows.append(
                        {
                            "encoder": encoder,
                            "window_m": window_m,
                            "policy": policy,
                            "train_tier": tier,
                            "split": split,
                            "l2": l2,
                            "n_train": len(train_rows),
                            "n_eval": len(eval_rows),
                            **metrics,
                        }
                    )
                    counts = per_label_counts(y_true, y_pred, labels)
                    for label, values in counts.items():
                        precision = (
                            values["true_positive"]
                            / (values["true_positive"] + values["false_positive"])
                            if values["true_positive"] + values["false_positive"]
                            else 0.0
                        )
                        recall = (
                            values["true_positive"]
                            / (values["true_positive"] + values["false_negative"])
                            if values["true_positive"] + values["false_negative"]
                            else 0.0
                        )
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if precision + recall
                            else 0.0
                        )
                        per_class_rows.append(
                            {
                                "encoder": encoder,
                                "window_m": window_m,
                                "policy": policy,
                                "split": split,
                                "l2": l2,
                                "label": label,
                                "support": values["support"],
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                            }
                        )
                    for pageid, true, pred in zip(eval_rows["pageid"], y_true, y_pred, strict=True):
                        prediction_rows.append(
                            {
                                "encoder": encoder,
                                "window_m": window_m,
                                "policy": policy,
                                "split": split,
                                "l2": l2,
                                "pageid": pageid,
                                "label": true,
                                "prediction": pred,
                            }
                        )
                    interval = _bootstrap_interval(
                        y_true,
                        y_pred,
                        labels,
                        seed=seed,
                        n_bootstrap=n_bootstrap,
                    )
                    bootstrap_rows.append(
                        {
                            "encoder": encoder,
                            "window_m": window_m,
                            "policy": policy,
                            "split": split,
                            "l2": l2,
                            **interval,
                        }
                    )
                    matrices[row_id] = {
                        "labels": labels,
                        "matrix": _confusion_matrix(y_true, y_pred, labels),
                    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    write_csv_atomic(output_dir / "weighted_probe_metrics.csv", metrics, index=False)
    write_csv_atomic(
        output_dir / "weighted_probe_predictions.csv", pd.DataFrame(prediction_rows), index=False
    )
    write_csv_atomic(
        output_dir / "per_class_metrics.csv", pd.DataFrame(per_class_rows), index=False
    )
    write_csv_atomic(
        output_dir / "bootstrap_confidence_intervals.csv",
        pd.DataFrame(bootstrap_rows),
        index=False,
    )
    write_json_atomic(output_dir / "confusion_matrices.json", matrices)
    write_json_atomic(
        output_dir / "run_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "splits_path": str(splits_path),
            "weights_path": str(weights_path),
            "embeddings_paths": [str(path) for path in embeddings_paths],
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2_grid": list(L2_GRID),
            "headline_metrics": ["balanced_accuracy_supported", "macro_f1_supported"],
            "continuity_metrics": ["balanced_accuracy_allowed", "macro_f1_allowed"],
        },
    )
    if metrics.empty:
        summary = "# quality_weighted_multiscale_image_probe_v1\n\nNo metrics were produced.\n"
    else:
        best = metrics.sort_values("balanced_accuracy_supported", ascending=False).head(10)
        summary = (
            "# quality_weighted_multiscale_image_probe_v1\n\n"
            "Headline metric: `balanced_accuracy_supported`. Allowed-label metrics are retained "
            "for continuity with prior experiments.\n\n"
            + markdown_table(rows=best.to_dict("records"))
            + "\n"
        )
    write_text_atomic(output_dir / "summary.md", summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=experiment_artifact_dir(EXPERIMENT_ID))
    parser.add_argument("--splits-path", type=Path)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--embeddings-path", type=Path, action="append")
    parser.add_argument("--encoders", default="clip_base")
    parser.add_argument("--windows", default="320,2240")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    embeddings_paths = args.embeddings_path or embedding_cache_paths(
        output_dir,
        encoders=parse_csv_strings(args.encoders),
        windows=parse_csv_strings(args.windows),
    )
    run_probe(
        splits_path=args.splits_path or image_probe_splits_path(output_dir),
        weights_path=args.weights_path or sample_weights_path(output_dir),
        embeddings_paths=embeddings_paths,
        output_dir=output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
