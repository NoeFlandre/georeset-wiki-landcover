"""Shared quality/relevance/spatial subset masks for text-source analyses."""

from __future__ import annotations

import pandas as pd


def quality_subset_masks(records: pd.DataFrame) -> dict[str, pd.Series]:
    empty = pd.Series("", index=records.index)
    relevance = records.get("landcover_relevance", empty).astype("string").str.lower()
    spatial = pd.to_numeric(records.get("point_label_share_250m", empty), errors="coerce")
    quality_bin = records.get("quality_bin", empty).astype("string")
    recommended_use = records.get("recommended_use", empty).astype("string")
    high_quality = quality_bin.isin(["quality_high", "quality_very_high"])
    spatial_high = spatial >= 0.8
    relevance_medium_high = relevance.isin(["medium", "high"])
    return {
        "all": pd.Series(True, index=records.index),
        "relevance_medium_high": relevance_medium_high,
        "spatial_250m_ge_0.8": spatial_high,
        "relevance_medium_high_and_spatial_250m_ge_0.8": relevance_medium_high & spatial_high,
        "quality_high_or_very_high": high_quality,
        "quality_high_or_very_high_and_spatial_250m_ge_0.8": high_quality & spatial_high,
        "recommended_use_training": recommended_use == "use_for_training",
        "recommended_use_evaluation_only": recommended_use == "use_for_evaluation_only",
        "recommended_use_exclude": recommended_use == "exclude",
    }
