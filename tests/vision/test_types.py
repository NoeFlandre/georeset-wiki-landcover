import numpy as np

from georeset_wiki_landcover.vision.types import normalize_features


def test_normalize_features_returns_float32_unit_rows_without_nan_for_zero_rows() -> None:
    features = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)

    normalized = normalize_features(features)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[0], np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(normalized[1], np.array([0.0, 0.0], dtype=np.float32))
