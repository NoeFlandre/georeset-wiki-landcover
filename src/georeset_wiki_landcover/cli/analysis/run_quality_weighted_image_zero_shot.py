"""Run Experiment 014 zero-shot CLIP baselines for cached multiscale embeddings."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from georeset_wiki_landcover.analysis.supported_metrics import single_label_metrics_supported
from georeset_wiki_landcover.cli.analysis.run_clip_zero_shot_experiment import (
    build_transformers_clip_text_encoder,
)
from georeset_wiki_landcover.cli.csv_args import parse_csv_strings
from georeset_wiki_landcover.cli.image_probe_args import (
    embedding_cache_paths,
    image_probe_splits_path,
)
from georeset_wiki_landcover.experiment_paths import experiment_artifact_dir
from georeset_wiki_landcover.utils.json_io import (
    markdown_table,
    write_csv_atomic,
    write_text_atomic,
)
from georeset_wiki_landcover.vision.clip_embedding_cache import (
    load_embedding_cache,
    stack_embeddings_for_rows,
)
from georeset_wiki_landcover.vision.clip_zero_shot import (
    TextEncoder,
    build_corine_zero_shot_prompts,
    embed_label_prompts,
    predict_zero_shot,
)

EXPERIMENT_ID = "quality_weighted_multiscale_image_probe_v1"
TextEncoderFactory = Callable[[str, str], TextEncoder]


def zero_shot_image_probe_metrics_path(output_dir: Path) -> Path:
    return output_dir / "zero_shot_image_probe_metrics.csv"


def zero_shot_image_probe_predictions_path(output_dir: Path) -> Path:
    return output_dir / "zero_shot_image_probe_predictions.csv"


def zero_shot_image_probe_summary_path(output_dir: Path) -> Path:
    return output_dir / "zero_shot_image_probe_summary.md"


def _is_clip_encoder(encoder_name: str) -> bool:
    return encoder_name.startswith("clip_")


def run_zero_shot_image_probe(
    *,
    splits_path: Path,
    embeddings_paths: list[Path],
    output_dir: Path,
    device: str,
    text_encoder_factory: TextEncoderFactory | None = None,
) -> None:
    if text_encoder_factory is None:

        def default_text_encoder_factory(model_name: str, device: str) -> TextEncoder:
            return build_transformers_clip_text_encoder(model_name=model_name, device=device)

        text_encoder_factory = default_text_encoder_factory
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    eval_rows = splits[splits["split"] == "eval_strict"].copy()
    labels = sorted(splits["label"].dropna().astype(str).unique().tolist())
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for embeddings_path in embeddings_paths:
        if not embeddings_path.exists():
            continue
        data = np.load(embeddings_path)
        encoder = str(data["encoder_name"]) if "encoder_name" in data else embeddings_path.stem
        if not _is_clip_encoder(encoder):
            continue
        model_name = (
            str(data["model_name"]) if "model_name" in data else "openai/clip-vit-base-patch32"
        )
        window_m = int(data["window_m"]) if "window_m" in data else -1
        embeddings = load_embedding_cache(embeddings_path)
        rows, eval_x = stack_embeddings_for_rows(
            eval_rows, embeddings, context=f"{encoder}/zero_shot"
        )
        y_true = rows["label"].to_numpy(dtype=str)
        prompts = build_corine_zero_shot_prompts(labels)
        text_encoder = text_encoder_factory(model_name, device)
        text_embeddings = embed_label_prompts(prompts, text_encoder)
        y_pred = predict_zero_shot(eval_x.astype(np.float32), text_embeddings)
        metrics = single_label_metrics_supported(y_true, y_pred, labels)
        metric_rows.append(
            {
                "baseline": "zero_shot_clip",
                "encoder": encoder,
                "model_name": model_name,
                "window_m": window_m,
                "n_eval": len(rows),
                "n_labels": len(prompts),
                **metrics,
            }
        )
        prediction_rows.extend(
            {
                "baseline": "zero_shot_clip",
                "encoder": encoder,
                "window_m": window_m,
                "pageid": pageid,
                "label": target,
                "prediction": prediction,
            }
            for pageid, target, prediction in zip(
                rows["pageid"].astype(str), y_true, y_pred, strict=True
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    write_csv_atomic(zero_shot_image_probe_metrics_path(output_dir), metrics, index=False)
    write_csv_atomic(
        zero_shot_image_probe_predictions_path(output_dir),
        pd.DataFrame(prediction_rows),
        index=False,
    )
    if metrics.empty:
        summary = "# zero_shot_image_probe\n\nNo CLIP embedding files were available.\n"
    else:
        summary = (
            "# zero_shot_image_probe\n\n"
            "Zero-shot CLIP baselines for Experiment 014 multiscale image embeddings.\n\n"
            + markdown_table(rows=metrics.to_dict("records"))
            + "\n"
        )
    write_text_atomic(zero_shot_image_probe_summary_path(output_dir), summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=experiment_artifact_dir(EXPERIMENT_ID))
    parser.add_argument("--splits-path", type=Path)
    parser.add_argument("--embeddings-path", type=Path, action="append")
    parser.add_argument("--encoders", default="clip_base")
    parser.add_argument("--windows", default="320,2240")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    embeddings_paths = args.embeddings_path or embedding_cache_paths(
        output_dir,
        encoders=parse_csv_strings(args.encoders),
        windows=parse_csv_strings(args.windows),
    )
    run_zero_shot_image_probe(
        splits_path=args.splits_path or image_probe_splits_path(output_dir),
        embeddings_paths=embeddings_paths,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
