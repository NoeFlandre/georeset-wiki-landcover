"""Small NumPy softmax linear probe for frozen image embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]
LinearProbeField = Literal["labels", "weights", "bias", "mean", "scale"]


@dataclass(frozen=True)
class LinearProbeModel:
    labels: StringArray
    weights: FloatArray
    bias: FloatArray
    mean: FloatArray
    scale: FloatArray

    def __getitem__(self, key: LinearProbeField) -> StringArray | FloatArray:
        if key == "labels":
            return self.labels
        if key == "weights":
            return self.weights
        if key == "bias":
            return self.bias
        if key == "mean":
            return self.mean
        return self.scale


def _standardize(features: NDArray[np.float32] | FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    return (features - mean) / scale, mean, scale


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float64)


def fit_linear_probe(
    features: NDArray[np.float32] | FloatArray,
    labels: StringArray,
    *,
    seed: int,
    epochs: int = 500,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> LinearProbeModel:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

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
        diff = (probs - target) / len(x)
        weights -= learning_rate * (x.T @ diff + l2 * weights)
        bias -= learning_rate * diff.sum(axis=0)
    return LinearProbeModel(
        labels=unique_labels,
        weights=weights,
        bias=bias,
        mean=mean,
        scale=scale,
    )


def predict_linear_probe(model: LinearProbeModel, features: NDArray[np.float32] | FloatArray) -> StringArray:
    x = (features.astype(np.float64) - model.mean) / model.scale
    indices = np.argmax(x @ model.weights + model.bias, axis=1)
    return np.asarray(model.labels[indices])


def evaluate_predictions(y_true: StringArray, y_pred: StringArray) -> dict[str, float]:
    labels = np.array(sorted(set(y_true.tolist()) | set(y_pred.tolist())))
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    recalls = []
    f1s = []
    for label in labels:
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }
