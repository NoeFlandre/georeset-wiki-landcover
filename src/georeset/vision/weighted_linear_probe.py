"""Weighted NumPy softmax linear probe for frozen image embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from georeset.vision.linear_probe import (
    FloatArray,
    LinearProbeModel,
    StringArray,
    _softmax,
    _standardize,
)


def fit_weighted_linear_probe(
    features: NDArray[np.float32] | FloatArray,
    labels: StringArray,
    sample_weight: NDArray[np.float64] | FloatArray,
    *,
    seed: int,
    epochs: int = 500,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> LinearProbeModel:
    if len(features) != len(labels) or len(labels) != len(sample_weight):
        raise ValueError("features, labels, and sample_weight must have the same number of rows")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    weights_vector = np.asarray(sample_weight, dtype=np.float64)
    if np.any(weights_vector < 0):
        raise ValueError("sample weights must be non-negative")
    weight_sum = float(weights_vector.sum())
    if weight_sum <= 0.0:
        raise ValueError("sample weights must have positive sum")

    unique_labels = np.array(sorted(set(labels.tolist())))
    if len(unique_labels) < 2:
        raise ValueError("linear probe needs at least two labels")
    x, mean, scale = _standardize(features.astype(np.float64))
    y = np.array([int(np.where(unique_labels == label)[0][0]) for label in labels])
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], len(unique_labels)))
    bias = np.zeros(len(unique_labels), dtype=np.float64)
    target = np.eye(len(unique_labels))[y]
    for _ in range(epochs):
        probs = _softmax(x @ weights + bias)
        diff = (probs - target) * weights_vector[:, None] / weight_sum
        weights -= learning_rate * (x.T @ diff + l2 * weights)
        bias -= learning_rate * diff.sum(axis=0)
    return LinearProbeModel(
        labels=unique_labels,
        weights=weights,
        bias=bias,
        mean=mean,
        scale=scale,
    )
