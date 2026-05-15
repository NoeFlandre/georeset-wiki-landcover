from pathlib import Path

import numpy as np

from georeset.vision.clip_embedding_cache import load_embedding_cache


def test_load_embedding_cache_returns_string_pageid_float32_mapping(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        pageids=np.array([1, "2"]),
        embeddings=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    )

    embeddings = load_embedding_cache(path)

    assert list(embeddings) == ["1", "2"]
    assert embeddings["1"].dtype == np.float32
    assert embeddings["2"].tolist() == [3.0, 4.0]
