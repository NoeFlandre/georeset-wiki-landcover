"""Shared helpers for CLIP embedding cache artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from georeset_wiki_landcover.vision.types import FloatFeatures


def load_embedding_cache(path: Path) -> dict[str, FloatFeatures]:
    data = np.load(path)
    pageids = data["pageids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    if len(pageids) != len(embeddings):
        raise ValueError("pageids and embeddings must have the same number of rows")

    return dict(zip(pageids.tolist(), embeddings, strict=True))


def stack_embeddings_for_rows(
    rows: pd.DataFrame,
    embeddings: dict[str, FloatFeatures],
    *,
    context: str,
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, FloatFeatures]:
    pageids = rows["pageid"].astype(str)
    missing = sorted(set(pageids) - set(embeddings))
    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(f"Missing cached embeddings for {context}: {preview}{suffix}")
    filtered = rows[pageids.isin(embeddings)].copy()
    if filtered.empty:
        raise ValueError(f"No cached embeddings available for {context}.")
    matrix = np.stack([embeddings[str(pageid)] for pageid in filtered["pageid"]]).astype(np.float32)
    return filtered, matrix
