"""Build split and sample-weight metadata for quality-weighted image probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from georeset_wiki_landcover.utils.boolish import parse_boolish_series

TRAIN_TIERS = ("all", "spatial_only", "quality_spatial", "text_spatial_agreement")
RELEVANCE_COMPONENT = {"none": 0.25, "low": 0.60, "medium": 1.00, "high": 1.25}
UNCERTAINTY_COMPONENT = {"high": 0.40, "medium": 0.75, "low": 1.00}


def load_ok_predictions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(pageid): str(record["prediction"])
        for pageid, record in data.items()
        if isinstance(record, dict)
        and record.get("parse_status") == "ok"
        and record.get("prediction") is not None
    }


def _component_from_map(values: pd.Series, mapping: dict[str, float], missing: float) -> pd.Series:
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.map(mapping).fillna(missing).astype(float)


def compute_quality_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach raw and class-balanced quality weights to one row per pageid."""
    weighted = frame.copy()
    relevance = _component_from_map(weighted["landcover_relevance"], RELEVANCE_COMPONENT, 0.50)
    uncertainty = _component_from_map(weighted["uncertainty"], UNCERTAINTY_COMPONENT, 0.75)
    spatial = pd.to_numeric(weighted["point_label_share_250m"], errors="coerce").clip(0.20, 1.00)
    spatial = spatial.fillna(0.20).astype(float)
    qwen_correct = weighted["qwen_prediction"].astype(str).eq(weighted["label"].astype(str))
    gemma_correct = weighted["gemma_prediction"].astype(str).eq(weighted["label"].astype(str))
    qwen_gemma_agree = (
        weighted["qwen_prediction"].astype(str).eq(weighted["gemma_prediction"].astype(str))
    )
    agreement = np.select(
        [
            qwen_correct & gemma_correct,
            qwen_correct ^ gemma_correct,
            qwen_gemma_agree & ~(qwen_correct | gemma_correct),
        ],
        [1.25, 1.00, 0.75],
        default=0.85,
    )
    weighted["relevance_component"] = relevance
    weighted["uncertainty_component"] = uncertainty
    weighted["spatial_component"] = spatial
    weighted["agreement_component"] = agreement.astype(float)
    raw = relevance * uncertainty * spatial * weighted["agreement_component"]
    weighted["weight_raw"] = _normalize_mean_one(raw)
    class_mean = weighted.groupby("label")["weight_raw"].transform("mean")
    weighted["weight_class_balanced"] = _normalize_mean_one(weighted["weight_raw"] / class_mean)
    return weighted


def _normalize_mean_one(values: pd.Series) -> pd.Series:
    mean = float(values.mean()) if len(values) else 1.0
    if mean <= 0:
        return pd.Series(1.0, index=values.index)
    return (values / mean).astype(float)


def add_geo_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic EPSG:2154 10km and 20km geographic group ids."""
    grouped = frame.copy()
    try:
        import geopandas as gpd

        points = gpd.GeoDataFrame(
            grouped,
            geometry=gpd.points_from_xy(grouped["lon"].astype(float), grouped["lat"].astype(float)),
            crs="EPSG:4326",
        ).to_crs("EPSG:2154")
        x = points.geometry.x
        y = points.geometry.y
        grouped["geo_group_10km"] = (
            (x // 10000).astype(int).astype(str) + "_" + (y // 10000).astype(int).astype(str)
        )
        grouped["geo_group_20km"] = (
            (x // 20000).astype(int).astype(str) + "_" + (y // 20000).astype(int).astype(str)
        )
    except Exception as exc:
        raise ValueError("Cannot compute EPSG:2154 geo groups from lon/lat columns") from exc
    return grouped


def _base_frame(
    quality_scores_path: Path, qwen_predictions_path: Path, gemma_predictions_path: Path
) -> pd.DataFrame:
    frame = pd.read_csv(quality_scores_path, dtype={"pageid": str, "corine_label": str})
    frame = frame.rename(columns={"corine_label": "label"}).copy()
    frame["pageid"] = frame["pageid"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame = frame[frame["label"].fillna("").astype(str).str.len() > 0].copy()
    qwen = load_ok_predictions(qwen_predictions_path)
    gemma = load_ok_predictions(gemma_predictions_path)
    frame["qwen_prediction"] = frame["pageid"].map(qwen)
    frame["gemma_prediction"] = frame["pageid"].map(gemma)
    frame["models_agree_with_label"] = frame["qwen_prediction"].eq(
        frame["gemma_prediction"]
    ) & frame["qwen_prediction"].eq(frame["label"])
    frame["dominant_matches_point_label_250m_parsed"] = parse_boolish_series(
        frame["dominant_matches_point_label_250m"]
    )
    frame["spatial_ok"] = (
        pd.to_numeric(frame["point_label_share_250m"], errors="coerce").ge(0.8)
        & frame["dominant_matches_point_label_250m_parsed"]
    )
    frame["spatial_only"] = frame["spatial_ok"]
    frame["relevance_medium_high"] = frame["landcover_relevance"].isin(["medium", "high"])
    frame["quality_high_or_very_high"] = frame["quality_bin"].isin(
        ["quality_high", "quality_very_high"]
    )
    frame["low_uncertainty"] = ~frame["uncertainty"].eq("high")
    frame["quality_spatial"] = (
        frame["spatial_ok"]
        & frame["relevance_medium_high"]
        & frame["quality_high_or_very_high"]
        & frame["low_uncertainty"]
    )
    frame["text_spatial_agreement"] = frame["quality_spatial"] & frame["models_agree_with_label"]
    frame = add_geo_groups(frame)
    return compute_quality_weights(frame)


def _tier_mask(frame: pd.DataFrame, tier: str) -> pd.Series:
    if tier == "all":
        return pd.Series(True, index=frame.index)
    if tier in {"spatial_only", "quality_spatial", "text_spatial_agreement"}:
        return parse_boolish_series(frame[tier])
    raise ValueError(f"Unknown image probe tier: {tier}")


def tier_mask(frame: pd.DataFrame, tier: str) -> pd.Series:
    """Return a boolean mask for an image-probe train tier."""
    return _tier_mask(frame, tier)


def _sample_eval(frame: pd.DataFrame, *, seed: int, eval_per_class: int) -> pd.DataFrame:
    rng_seed = seed
    sampled = []
    for _, group in frame.sort_values(["label", "pageid"]).groupby("label", sort=True):
        sampled.append(group.sample(n=min(len(group), eval_per_class), random_state=rng_seed))
    if not sampled:
        return frame.iloc[0:0].copy()
    return (
        pd.concat(sampled, ignore_index=True)
        .sort_values(["label", "pageid"])
        .reset_index(drop=True)
    )


def _spatial_block_splits(frame: pd.DataFrame, *, n_folds: int = 5) -> pd.DataFrame:
    unique_groups = sorted(frame["geo_group_20km"].astype(str).unique().tolist())
    if not unique_groups:
        return frame.iloc[0:0].copy()
    group_sizes = frame.groupby("geo_group_20km").size().sort_values(ascending=False)
    fold_sizes = [0 for _ in range(n_folds)]
    group_to_fold: dict[str, int] = {}
    for group in group_sizes.index.astype(str):
        fold = min(range(n_folds), key=lambda index: (fold_sizes[index], index))
        group_to_fold[group] = fold
        fold_sizes[fold] += int(group_sizes.loc[group])
    rows = []
    for fold in range(n_folds):
        fold_rows = frame[frame["geo_group_20km"].astype(str).map(group_to_fold).eq(fold)].copy()
        fold_rows["split"] = f"spatial_block_fold_{fold}"
        fold_rows["tier"] = "eval_spatial_block"
        rows.append(fold_rows)
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def _train_rows_for_eval_split(
    base: pd.DataFrame,
    *,
    eval_rows: pd.DataFrame,
    split_name: str,
) -> list[pd.DataFrame]:
    eval_pageids = set(eval_rows["pageid"].astype(str))
    train_base = base[~base["pageid"].astype(str).isin(eval_pageids)].copy()
    rows: list[pd.DataFrame] = []
    for tier in TRAIN_TIERS:
        tier_rows = train_base[_tier_mask(train_base, tier)].copy()
        tier_rows["split"] = f"train_for_{split_name}"
        tier_rows["tier"] = tier
        rows.append(tier_rows)
    return rows


def build_image_probe_splits_v2(
    *,
    quality_scores_path: Path,
    qwen_predictions_path: Path,
    gemma_predictions_path: Path,
    seed: int = 42,
    eval_per_class: int = 5,
    n_repeated_splits: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if eval_per_class <= 0:
        raise ValueError("eval_per_class must be positive")
    if n_repeated_splits < 0:
        raise ValueError("n_repeated_splits must be non-negative")
    base = _base_frame(quality_scores_path, qwen_predictions_path, gemma_predictions_path)
    eval_pool = base[base["quality_spatial"]].copy()
    eval_strict = _sample_eval(eval_pool, seed=seed, eval_per_class=eval_per_class)
    eval_pageids = set(eval_strict["pageid"])

    split_rows = []
    eval_rows = eval_strict.copy()
    eval_rows["split"] = "eval_strict"
    eval_rows["tier"] = "eval_strict"
    split_rows.append(eval_rows)
    train_base = base[~base["pageid"].isin(eval_pageids)].copy()
    for tier in TRAIN_TIERS:
        rows = train_base[_tier_mask(train_base, tier)].copy()
        rows["split"] = "train"
        rows["tier"] = tier
        split_rows.append(rows)
    split_rows.extend(
        _train_rows_for_eval_split(base, eval_rows=eval_strict, split_name="eval_strict")
    )
    for repeat in range(n_repeated_splits):
        rows = _sample_eval(eval_pool, seed=seed + repeat, eval_per_class=eval_per_class).copy()
        rows["split"] = f"repeated_eval_seed_{repeat}"
        rows["tier"] = "eval_repeated"
        split_rows.append(rows)
        split_rows.extend(
            _train_rows_for_eval_split(
                base, eval_rows=rows, split_name=f"repeated_eval_seed_{repeat}"
            )
        )
    spatial_rows = _spatial_block_splits(base)
    split_rows.append(spatial_rows)
    for split_name, rows in spatial_rows.groupby("split", sort=True):
        split_rows.extend(
            _train_rows_for_eval_split(base, eval_rows=rows, split_name=str(split_name))
        )
    splits = pd.concat(split_rows, ignore_index=True)
    weights = base[
        [
            "pageid",
            "label",
            "relevance_component",
            "uncertainty_component",
            "spatial_component",
            "agreement_component",
            "weight_raw",
            "weight_class_balanced",
        ]
    ].copy()
    manifest: dict[str, Any] = {
        "experiment_id": "quality_weighted_multiscale_image_probe_v1",
        "seed": seed,
        "eval_per_class": eval_per_class,
        "n_repeated_splits": n_repeated_splits,
        "train_tiers": list(TRAIN_TIERS),
        "evaluation_pool": "quality_spatial",
        "evaluation_selection_excludes_model_agreement": True,
        "split_specific_train_rows": True,
        "split_specific_train_rule": "train_for_<eval_split> excludes that eval split's pageids",
        "n_rows": int(len(base)),
        "n_eval_strict": int(len(eval_strict)),
    }
    return splits.reset_index(drop=True), weights.reset_index(drop=True), manifest
