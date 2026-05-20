"""Small helpers for tabular artifacts keyed by Wikipedia pageid."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_optional_pageid_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["pageid"])
    frame = pd.read_csv(path, dtype={"pageid": str})
    if "pageid" not in frame.columns:
        return pd.DataFrame(columns=["pageid"])
    frame["pageid"] = frame["pageid"].astype(str)
    return frame


def dataframe_by_pageid(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "pageid" not in frame.columns:
        return {}
    indexed: dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        indexed[str(row["pageid"])] = row
    return indexed
