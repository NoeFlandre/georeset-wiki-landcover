"""Random training-set controls for Experiment 014 hard-filter image probes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from georeset.analysis.supported_metrics import single_label_metrics_supported
from georeset.cli.csv_args import parse_csv_strings
from georeset.cli.image_probe_args import embedding_cache_paths, image_probe_splits_path
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
HARD_TIERS = ("quality_spatial", "text_spatial_agreement")


def image_probe_random_controls_path(output_dir: Path) -> Path:
    return output_dir / "image_probe_random_training_controls.csv"


def image_probe_random_controls_markdown_path(output_dir: Path) -> Path:
    return output_dir / "image_probe_random_training_controls.md"


def image_probe_control_manifest_path(output_dir: Path) -> Path:
    return output_dir / "control_manifest.json"


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *[str(part) for part in parts]]).encode()
    ).hexdigest()
    return int(digest[:16], 16) % (2**32)


def random_same_n(pageids: list[str], n: int, *, seed: int) -> list[str]:
    if n > len(pageids):
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(pageids, dtype=str), size=n, replace=False).tolist())


def random_same_target_distribution(
    universe: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for label, group in observed.groupby("label", sort=True):
        candidates = universe[universe["label"].eq(label)]["pageid"].astype(str).to_numpy()
        n = len(group)
        if n > len(candidates):
            return []
        selected.extend(rng.choice(candidates, size=n, replace=False).astype(str).tolist())
    return sorted(selected)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "random_mean": np.nan,
            "random_std": np.nan,
            "random_p02_5": np.nan,
            "random_p50": np.nan,
            "random_p97_5": np.nan,
        }
    array = np.asarray(values, dtype=float)
    return {
        "random_mean": float(array.mean()),
        "random_std": float(array.std(ddof=0)),
        "random_p02_5": float(np.quantile(array, 0.025)),
        "random_p50": float(np.quantile(array, 0.5)),
        "random_p97_5": float(np.quantile(array, 0.975)),
    }


def _evaluate_subset(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    embeddings: dict[str, NDArray[np.float32]],
    labels: list[str],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> float:
    if train_rows["label"].nunique() < 2:
        return float("nan")
    train_rows, train_x = stack_embeddings_for_rows(train_rows, embeddings, context="control train")
    eval_rows, eval_x = stack_embeddings_for_rows(eval_rows, embeddings, context="control eval")
    model = fit_weighted_linear_probe(
        train_x,
        train_rows["label"].to_numpy(dtype=str),
        np.ones(len(train_rows), dtype=np.float64),
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    pred = predict_linear_probe(model, eval_x)
    metrics = single_label_metrics_supported(eval_rows["label"].to_numpy(dtype=str), pred, labels)
    return metrics["balanced_accuracy_supported"]


def evaluate_controls(
    *,
    splits_path: Path,
    embeddings_paths: list[Path],
    output_dir: Path,
    n_draws: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> None:
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    labels = sorted(splits["label"].astype(str).unique().tolist())
    rows_out: list[dict[str, Any]] = []
    for embeddings_path in embeddings_paths:
        if not embeddings_path.exists():
            continue
        data = np.load(embeddings_path)
        encoder = str(data["encoder_name"]) if "encoder_name" in data else embeddings_path.stem
        window_m = int(data["window_m"]) if "window_m" in data else -1
        embeddings = load_embedding_cache(embeddings_path)
        universe = splits[(splits["split"] == "train") & (splits["tier"] == "all")].copy()
        eval_rows = splits[splits["split"] == "eval_strict"].copy()
        for tier in HARD_TIERS:
            observed = splits[(splits["split"] == "train") & (splits["tier"] == tier)].copy()
            if observed.empty:
                continue
            observed_score = _evaluate_subset(
                observed,
                eval_rows,
                embeddings,
                labels,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            )
            for control_type in ("same_n", "same_target_distribution"):
                scores: list[float] = []
                for draw in range(n_draws):
                    draw_seed = stable_seed(seed, encoder, window_m, tier, control_type, draw)
                    if control_type == "same_n":
                        pageids = random_same_n(
                            universe["pageid"].astype(str).tolist(),
                            len(observed),
                            seed=draw_seed,
                        )
                    else:
                        pageids = random_same_target_distribution(
                            universe,
                            observed,
                            seed=draw_seed,
                        )
                    if not pageids:
                        continue
                    score = _evaluate_subset(
                        universe[universe["pageid"].isin(pageids)],
                        eval_rows,
                        embeddings,
                        labels,
                        seed=draw_seed,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        l2=l2,
                    )
                    if not np.isnan(score):
                        scores.append(score)
                rows_out.append(
                    {
                        "encoder": encoder,
                        "window_m": window_m,
                        "tier": tier,
                        "control_type": control_type,
                        "observed_score": observed_score,
                        "n_train_observed": len(observed),
                        "n_draws_requested": n_draws,
                        "n_draws_successful": len(scores),
                        **_summary(scores),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = pd.DataFrame(rows_out)
    write_csv_atomic(image_probe_random_controls_path(output_dir), controls, index=False)
    write_text_atomic(
        image_probe_random_controls_markdown_path(output_dir),
        "# Experiment 014 random training controls\n\n"
        + (
            markdown_table(rows=controls.to_dict("records"))
            if not controls.empty
            else "No controls produced."
        )
        + "\n",
    )
    write_json_atomic(
        image_probe_control_manifest_path(output_dir),
        {
            "experiment_id": EXPERIMENT_ID,
            "splits_path": str(splits_path),
            "embeddings_paths": [str(path) for path in embeddings_paths],
            "n_draws": n_draws,
            "seed": seed,
            "controls": ["same_n", "same_target_distribution"],
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=experiment_artifact_dir(EXPERIMENT_ID))
    parser.add_argument("--splits-path", type=Path)
    parser.add_argument("--embeddings-path", type=Path, action="append")
    parser.add_argument("--encoders", default="clip_base")
    parser.add_argument("--windows", default="320,2240")
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    embeddings_paths = args.embeddings_path or embedding_cache_paths(
        output_dir,
        encoders=parse_csv_strings(args.encoders),
        windows=parse_csv_strings(args.windows),
    )
    evaluate_controls(
        splits_path=args.splits_path or image_probe_splits_path(output_dir),
        embeddings_paths=embeddings_paths,
        output_dir=output_dir,
        n_draws=args.n_draws,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )


if __name__ == "__main__":
    main()
