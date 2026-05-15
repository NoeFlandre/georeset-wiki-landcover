"""Shared helpers for CLIP embedding cache artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatFeatures = NDArray[np.float32]


def load_embedding_cache(path: Path) -> dict[str, FloatFeatures]:
    data = np.load(path)
    pageids = data["pageids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    return dict(zip(pageids.tolist(), embeddings, strict=True))


def stack_embeddings_for_rows(
    rows: pd.DataFrame,
    embeddings: dict[str, FloatFeatures],
    *,
    context: str,
) -> tuple[pd.DataFrame, FloatFeatures]:
    filtered = rows[rows["pageid"].isin(embeddings)].copy()
    if filtered.empty:
        raise ValueError(f"No cached embeddings available for {context}.")
    matrix = np.stack([embeddings[pageid] for pageid in filtered["pageid"]]).astype(np.float32)
    return filtered, matrix
