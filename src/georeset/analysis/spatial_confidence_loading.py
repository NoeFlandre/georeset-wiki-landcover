"""Utilities for loading spatial-confidence artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from georeset.utils.boolish import parse_boolish


def _coerce_dominant_match_value(value: object) -> object:
    parsed = parse_boolish(value)
    if parsed is None:
        return pd.NA
    return parsed


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
        raise ValueError(f"Spatial-confidence file '{path}' is missing required column: pageid.")

    df["pageid"] = df["pageid"].astype(str)
    if "point_label" in df.columns:
        df["point_label"] = df["point_label"].astype(str)
    if coerce_dominant_match_columns:
        df = _coerce_dominant_match_columns(df)
    return df
