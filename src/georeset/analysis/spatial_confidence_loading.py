"""Utilities for loading spatial-confidence artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_TRUE_VALUES = {"1", "t", "true", "yes", "y"}
_FALSE_VALUES = {"0", "f", "false", "no", "n", ""}
_NULL_VALUES = {"", "none", "null", "nan", "na"}


def _coerce_dominant_match_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return pd.NA
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    if normalized in _NULL_VALUES:
        return pd.NA
    return pd.NA


def _coerce_dominant_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if not column.startswith("dominant_matches_point_label_"):
            continue
        df[column] = df[column].map(_coerce_dominant_match_value).astype("boolean")
    return df


def load_spatial_confidence(
    path: Path,
    *,
    allow_missing_pageid: bool = False,
    coerce_dominant_match_columns: bool = True,
) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"pageid": str, "point_label": str})

    if "pageid" not in df.columns:
        if allow_missing_pageid:
            return pd.DataFrame()
        raise ValueError(
            f"Spatial-confidence file '{path}' is missing required column: pageid."
        )

    df["pageid"] = df["pageid"].astype(str)
    if "point_label" in df.columns:
        df["point_label"] = df["point_label"].astype(str)
    if coerce_dominant_match_columns:
        df = _coerce_dominant_match_columns(df)
    return df
