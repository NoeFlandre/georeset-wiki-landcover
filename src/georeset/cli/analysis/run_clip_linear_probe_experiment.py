"""Train CLIP linear probes from cached embeddings and split tiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from georeset.utils.json_io import write_csv_atomic, write_text_atomic
from georeset.vision.clip_embedding_cache import load_embedding_cache, stack_embeddings_for_rows
from georeset.vision.linear_probe import (
    evaluate_predictions,
    fit_linear_probe,
    predict_linear_probe,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, required=True)
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/clip_linear_probe_weak_labels_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    return parser.parse_args(argv)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows.\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def run_experiment(
    *,
    splits_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> None:
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    embeddings = load_embedding_cache(embeddings_path)
    eval_rows = splits[splits["split"] == "eval_strict"].copy()
    eval_rows, eval_x = stack_embeddings_for_rows(eval_rows, embeddings, context="eval_strict")
    eval_y = eval_rows["label"].to_numpy()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for tier, train_rows in splits[splits["split"] == "train"].groupby("tier", sort=True):
        train_rows = train_rows[train_rows["pageid"].isin(embeddings)]
        if train_rows["label"].nunique() < 2:
            continue
        train_rows, train_x = stack_embeddings_for_rows(train_rows, embeddings, context=f"train/{tier}")
        train_y = train_rows["label"].to_numpy()
        model = fit_linear_probe(
            train_x,
            train_y,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        predictions = predict_linear_probe(model, eval_x)
        metrics = evaluate_predictions(eval_y, predictions)
        metric_rows.append(
            {
                "tier": tier,
                "n_train": len(train_rows),
                "n_eval": len(eval_rows),
                "n_train_labels": train_rows["label"].nunique(),
                **metrics,
            }
        )
        prediction_rows.extend(
            {
                "tier": tier,
                "pageid": pageid,
                "target": target,
                "prediction": prediction,
            }
            for pageid, target, prediction in zip(
                eval_rows["pageid"], eval_y, predictions, strict=True
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output_dir / "linear_probe_metrics.csv", pd.DataFrame(metric_rows), index=False)
    write_csv_atomic(
        output_dir / "linear_probe_predictions.csv",
        pd.DataFrame(prediction_rows),
        index=False,
    )
    summary = pd.DataFrame(metric_rows).sort_values("balanced_accuracy", ascending=False)
    write_text_atomic(
        output_dir / "summary.md",
        "# clip_linear_probe_weak_labels_v1\n\n" + _markdown_table(summary),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(
        splits_path=args.splits_path,
        embeddings_path=args.embeddings_path,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
