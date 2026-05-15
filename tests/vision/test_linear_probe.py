import numpy as np

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

