"""Shared typing primitives for image-patch pipelines."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

UInt8ImageBatch = NDArray[np.uint8]
FloatFeatures = NDArray[np.float32]


class PatchEncoder(Protocol):
    def __call__(self, batch: UInt8ImageBatch) -> FloatFeatures: ...


def normalize_features(features: object) -> FloatFeatures:
    array = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(array / norms, dtype=np.float32)
