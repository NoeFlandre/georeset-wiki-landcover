"""Shared helpers for CLIP embedding cache artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatFeatures = NDArray[np.float32]


def load_embedding_cache(path: Path) -> dict[str, FloatFeatures]:
    data = np.load(path)
    pageids = data["pageids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    return dict(zip(pageids.tolist(), embeddings, strict=True))
