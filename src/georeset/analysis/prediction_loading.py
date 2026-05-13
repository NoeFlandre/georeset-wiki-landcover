"""Helpers for loading frozen classification prediction records."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.utils.json_io import read_json_file


def prediction_identity(path: Path) -> tuple[str, str]:
    """Return the prediction task and text source from a filename."""
    if not path.name.endswith("_predictions.json"):
        raise ValueError(f"Unknown prediction file name: {path.name}")

    stem = path.name.removesuffix("_predictions.json")
    if stem.startswith("corine_level2_"):
        return "corine_level2", stem.removeprefix("corine_level2_")
    if stem.startswith("osm_"):
        return "osm", stem.removeprefix("osm_")
    raise ValueError(f"Unknown prediction file name: {path.name}")


def infer_model_from_metadata(
    metadata: Mapping[str, Any],
    parent_dir: Path,
    *,
    metadata_keys: Sequence[str] = ("model", "model_repo_id"),
) -> str:
    """Infer model name from prediction metadata or experiment path naming convention."""
    for key in metadata_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    if "__gemma4_31b_it_q4_0" in str(parent_dir):
        return "gemma-4-31B-it-Q4_0.gguf"
    return "Qwen3.6-27B-Q4_0.gguf"


def infer_model_for_records(
    records: pd.DataFrame,
    parent_dir: Path,
    *,
    metadata_keys: Sequence[str] = ("model", "model_repo_id"),
) -> str:
    """Infer model from the first usable metadata entry, otherwise fall back to directory."""
    for raw_metadata in records.get("metadata", pd.Series(dtype=object)):
        if isinstance(raw_metadata, Mapping):
            return infer_model_from_metadata(
                raw_metadata,
                parent_dir,
                metadata_keys=metadata_keys,
            )
    return infer_model_from_metadata({}, parent_dir, metadata_keys=metadata_keys)


def load_prediction_records(
    experiment_dir: Path,
    *,
    text_sources: Collection[str] | None = None,
    normalize_targets: bool = False,
    include_source_dir: bool = False,
) -> pd.DataFrame:
    """Load frozen prediction files from an experiment directory into a DataFrame."""
    rows: list[dict[str, Any]] = []
    for path in sorted(experiment_dir.glob("*_predictions.json")):
        task, text_source = prediction_identity(path)
        if text_sources is not None and text_source not in text_sources:
            continue
        predictions = read_json_file(path)
        if not isinstance(predictions, Mapping):
            continue
        for pageid, payload in predictions.items():
            if not isinstance(payload, Mapping):
                continue
            target = payload.get("target")
            prediction = payload.get("prediction")
            if normalize_targets:
                if isinstance(target, list):
                    target = [str(value) for value in target]
                elif target is not None:
                    target = str(target)
                if isinstance(prediction, list):
                    prediction = [str(value) for value in prediction]
            rows.append(
                {
                    "pageid": str(payload.get("pageid", pageid)),
                    "task": task,
                    "text_source": text_source,
                    "target": target,
                    "prediction": prediction,
                    "parse_status": payload.get("parse_status"),
                    "metadata": payload.get("metadata", {}),
                }
            )
    if include_source_dir:
        for row in rows:
            row["source_parent_experiment_dir"] = str(experiment_dir)
    return pd.DataFrame(rows)
