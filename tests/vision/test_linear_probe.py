import numpy as np
import pytest

from georeset.vision.linear_probe import fit_linear_probe, predict_linear_probe


def test_linear_probe_learns_separable_embeddings() -> None:
    train_x = np.array(
        [
            [2.0, 0.0],
            [1.8, 0.2],
            [0.0, 2.0],
            [0.2, 1.8],
            [-2.0, 0.0],
            [-1.8, -0.2],
        ],
        dtype=np.float32,
    )
    train_y = np.array(["forest", "forest", "water", "water", "crop", "crop"])

    model = fit_linear_probe(train_x, train_y, seed=7, epochs=300, learning_rate=0.2)
    predictions = predict_linear_probe(model, train_x)

    assert predictions.tolist() == train_y.tolist()
    assert model["weights"].shape == (2, 3)
    assert model["labels"].tolist() == ["crop", "forest", "water"]


def test_linear_probe_model_exposes_typed_attributes_and_legacy_mapping_access() -> None:
    train_x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    train_y = np.array(["forest", "water"])

    model = fit_linear_probe(train_x, train_y, seed=7, epochs=1)

    assert model.weights.shape == (2, 2)
    assert model["weights"] is model.weights


def test_linear_probe_rejects_non_positive_epochs() -> None:
    train_x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    train_y = np.array(["forest", "water"])

    with pytest.raises(ValueError, match="epochs must be positive"):
        fit_linear_probe(train_x, train_y, seed=7, epochs=0)
