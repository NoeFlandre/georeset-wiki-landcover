import numpy as np

from georeset_wiki_landcover.vision.linear_probe import fit_linear_probe, predict_linear_probe
from georeset_wiki_landcover.vision.weighted_linear_probe import fit_weighted_linear_probe


def test_weighted_linear_probe_matches_unweighted_with_uniform_weights() -> None:
    features = np.array([[2.0, 0.0], [1.8, 0.1], [0.0, 2.0], [0.1, 1.8]], dtype=np.float32)
    labels = np.array(["a", "a", "b", "b"])

    unweighted = fit_linear_probe(features, labels, seed=3, epochs=200, learning_rate=0.2)
    weighted = fit_weighted_linear_probe(
        features,
        labels,
        np.ones(len(labels), dtype=np.float64),
        seed=3,
        epochs=200,
        learning_rate=0.2,
        l2=1e-4,
    )

    assert (
        predict_linear_probe(weighted, features).tolist()
        == predict_linear_probe(unweighted, features).tolist()
    )


def test_weighted_linear_probe_prioritizes_high_weight_samples() -> None:
    features = np.array([[1.0, 0.0], [1.1, 0.0], [1.2, 0.0], [3.0, 0.0]], dtype=np.float32)
    labels = np.array(["a", "a", "a", "b"])
    weights = np.array([0.1, 0.1, 0.1, 8.0])

    model = fit_weighted_linear_probe(
        features,
        labels,
        weights,
        seed=9,
        epochs=400,
        learning_rate=0.2,
        l2=1e-4,
    )

    assert predict_linear_probe(model, np.array([[2.8, 0.0]], dtype=np.float32)).tolist() == ["b"]
