"""Zero-shot CLIP evaluation for CORINE Sentinel patch embeddings."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from georeset.classification.labels import CORINE_LEVEL2_DESCRIPTIONS
from georeset.utils.json_io import write_csv_atomic, write_text_atomic
from georeset.vision.clip_embedding_cache import load_embedding_cache, stack_embeddings_for_rows
from georeset.vision.linear_probe import evaluate_predictions

FloatFeatures = NDArray[np.float32]
TextEncoder = Callable[[list[str]], FloatFeatures]


def zero_shot_metrics_path(output_dir: Path) -> Path:
    return output_dir / "zero_shot_clip_metrics.csv"


def zero_shot_predictions_path(output_dir: Path) -> Path:
    return output_dir / "zero_shot_clip_predictions.csv"


def zero_shot_summary_path(output_dir: Path) -> Path:
    return output_dir / "zero_shot_clip_summary.md"


def _normalize(features: FloatFeatures) -> FloatFeatures:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(features / norms, dtype=np.float32)


def build_corine_zero_shot_prompts(labels: Iterable[str]) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    for label in labels:
        description = CORINE_LEVEL2_DESCRIPTIONS[str(label)].lower()
        prompts[str(label)] = [
            f"a satellite image of {description}",
            f"a Sentinel-2 satellite image of {description}",
            f"land cover class: {description}",
        ]
    return prompts


def embed_label_prompts(
    prompts_by_label: dict[str, list[str]],
    text_encoder: TextEncoder,
) -> dict[str, FloatFeatures]:
    output: dict[str, FloatFeatures] = {}
    for label, prompts in prompts_by_label.items():
        if not prompts:
            raise ValueError(f"label {label} must have at least one prompt")

        encoded = _normalize(text_encoder(prompts).astype(np.float32))
        output[label] = _normalize(encoded.mean(axis=0, keepdims=True))[0]
    return output


def predict_zero_shot(
    image_embeddings: FloatFeatures,
    text_embeddings: dict[str, FloatFeatures],
) -> NDArray[np.str_]:
    if not text_embeddings:
        raise ValueError("text embeddings must not be empty")

    labels = np.array(sorted(text_embeddings), dtype=np.str_)
    image_features = _normalize(image_embeddings.astype(np.float32))
    text_features = np.stack([text_embeddings[label] for label in labels]).astype(np.float32)
    text_features = _normalize(text_features)
    indices = np.argmax(image_features @ text_features.T, axis=1)
    return np.asarray(labels[indices], dtype=np.str_)


def run_zero_shot_evaluation(
    *,
    splits_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    text_encoder: TextEncoder,
) -> None:
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    embeddings = load_embedding_cache(embeddings_path)
    eval_rows = splits[splits["split"] == "eval_strict"].copy()
    eval_rows, eval_x = stack_embeddings_for_rows(eval_rows, embeddings, context="eval_strict")
    eval_y = eval_rows["label"].to_numpy()
    prompts = build_corine_zero_shot_prompts(sorted(set(eval_y.tolist())))
    text_embeddings = embed_label_prompts(prompts, text_encoder)
    predictions = predict_zero_shot(eval_x, text_embeddings)
    metrics = evaluate_predictions(eval_y, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_row = {
        "baseline": "zero_shot_clip",
        "n_eval": len(eval_rows),
        "n_labels": len(prompts),
        **metrics,
    }
    write_csv_atomic(zero_shot_metrics_path(output_dir), pd.DataFrame([metric_row]), index=False)
    prediction_rows = [
        {
            "baseline": "zero_shot_clip",
            "pageid": pageid,
            "target": target,
            "prediction": prediction,
        }
        for pageid, target, prediction in zip(eval_rows["pageid"], eval_y, predictions, strict=True)
    ]
    write_csv_atomic(
        zero_shot_predictions_path(output_dir),
        pd.DataFrame(prediction_rows),
        index=False,
    )
    write_text_atomic(
        zero_shot_summary_path(output_dir),
        "# zero_shot_clip\n\n"
        f"- n_eval: {metric_row['n_eval']}\n"
        f"- n_labels: {metric_row['n_labels']}\n"
        f"- accuracy: {metric_row['accuracy']}\n"
        f"- balanced_accuracy: {metric_row['balanced_accuracy']}\n"
        f"- macro_f1: {metric_row['macro_f1']}\n",
    )
