"""Build deterministic CORINE split tiers for CLIP linear-probe experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from georeset.utils.boolish import parse_boolish
from georeset.utils.json_io import write_csv_atomic

TRAIN_TIERS = (
    "all",
    "spatial_only",
    "quality_spatial",
    "text_spatial_agreement",
)


def _load_predictions(path: Path) -> dict[str, str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(pageid): record.get("prediction")
        for pageid, record in data.items()
        if isinstance(record, dict) and record.get("parse_status", "ok") == "ok"
    }


def _sample_per_class(
    frame: pd.DataFrame, *, per_class: int, seed: int, exclude_pageids: set[str] | None = None
) -> pd.DataFrame:
    if exclude_pageids:
        frame = frame[~frame["pageid"].isin(exclude_pageids)]
    if frame.empty:
        return frame.copy()
    sampled = []
    for _, group in frame.sort_values(["label", "pageid"]).groupby("label", sort=True):
        sampled.append(group.sample(n=min(len(group), per_class), random_state=seed))
    return (
        pd.concat(sampled, ignore_index=True)
        .sort_values(["label", "pageid"])
        .reset_index(drop=True)
    )


def _base_frame(
    *,
    quality_scores_path: Path,
    qwen_predictions_path: Path,
    gemma_predictions_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(quality_scores_path, dtype={"pageid": str, "corine_label": str})
    qwen = _load_predictions(qwen_predictions_path)
    gemma = _load_predictions(gemma_predictions_path)
    frame = frame.rename(columns={"corine_label": "label"}).copy()
    frame["pageid"] = frame["pageid"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame["qwen_prediction"] = frame["pageid"].map(qwen)
    frame["gemma_prediction"] = frame["pageid"].map(gemma)
    frame["models_agree_with_label"] = frame["qwen_prediction"].eq(
        frame["gemma_prediction"]
    ) & frame["qwen_prediction"].eq(frame["label"])
    return frame


def _tier_mask(frame: pd.DataFrame, tier: str) -> pd.Series:
    spatial = frame["point_label_share_250m"].ge(0.8) & frame[
        "dominant_matches_point_label_250m"
    ].map(lambda value: parse_boolish(value) is True)
    relevance = frame["landcover_relevance"].isin(["medium", "high"])
    quality = frame["quality_bin"].isin(["quality_high", "quality_very_high"])
    low_uncertainty = ~frame["uncertainty"].eq("high")
    if tier == "all":
        return pd.Series(True, index=frame.index)
    if tier == "spatial_only":
        return spatial
    if tier == "quality_spatial":
        return spatial & relevance & quality & low_uncertainty
    if tier == "text_spatial_agreement":
        return spatial & relevance & quality & low_uncertainty & frame["models_agree_with_label"]
    raise ValueError(f"Unknown CLIP label tier: {tier}")


def build_clip_label_splits(
    *,
    quality_scores_path: Path,
    qwen_predictions_path: Path,
    gemma_predictions_path: Path,
    seed: int,
    eval_per_class: int = 5,
    train_per_class: int = 80,
) -> pd.DataFrame:
    if eval_per_class <= 0 or train_per_class <= 0:
        raise ValueError("per-class counts must be positive")

    frame = _base_frame(
        quality_scores_path=quality_scores_path,
        qwen_predictions_path=qwen_predictions_path,
        gemma_predictions_path=gemma_predictions_path,
    )
    eval_pool = frame[_tier_mask(frame, "quality_spatial")].copy()
    eval_rows = _sample_per_class(eval_pool, per_class=eval_per_class, seed=seed)
    eval_rows["split"] = "eval_strict"
    eval_rows["tier"] = "eval_strict"
    eval_pageids = set(eval_rows["pageid"])
    outputs = [eval_rows]
    for tier in TRAIN_TIERS:
        tier_rows = _sample_per_class(
            frame[_tier_mask(frame, tier)].copy(),
            per_class=train_per_class,
            seed=seed,
            exclude_pageids=eval_pageids,
        )
        tier_rows["split"] = "train"
        tier_rows["tier"] = tier
        outputs.append(tier_rows)
    result = pd.concat(outputs, ignore_index=True)
    columns = [
        "pageid",
        "title",
        "lat",
        "lon",
        "label",
        "split",
        "tier",
        "quality_score",
        "quality_bin",
        "recommended_use",
        "landcover_relevance",
        "uncertainty",
        "point_label_share_250m",
        "dominant_matches_point_label_250m",
        "qwen_prediction",
        "gemma_prediction",
        "models_agree_with_label",
    ]
    return result[[column for column in columns if column in result.columns]].sort_values(
        ["split", "tier", "label", "pageid"]
    )


def write_clip_label_splits(
    *,
    output_path: Path,
    quality_scores_path: Path,
    qwen_predictions_path: Path,
    gemma_predictions_path: Path,
    seed: int,
    eval_per_class: int,
    train_per_class: int,
) -> pd.DataFrame:
    splits = build_clip_label_splits(
        quality_scores_path=quality_scores_path,
        qwen_predictions_path=qwen_predictions_path,
        gemma_predictions_path=gemma_predictions_path,
        seed=seed,
        eval_per_class=eval_per_class,
        train_per_class=train_per_class,
    )
    write_csv_atomic(output_path, splits, index=False)
    return splits
