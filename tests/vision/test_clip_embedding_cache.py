from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from georeset.vision.clip_embedding_cache import load_embedding_cache, stack_embeddings_for_rows


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


def test_load_embedding_cache_rejects_mismatched_rows(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        pageids=np.array(["1", "2"]),
        embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
    )

    with pytest.raises(
        ValueError, match="pageids and embeddings must have the same number of rows"
    ):
        load_embedding_cache(path)


def test_stack_embeddings_for_rows_rejects_missing_pageids_by_default() -> None:
    rows = pd.DataFrame(
        [
            {"pageid": "1", "label": "31"},
            {"pageid": "missing", "label": "22"},
        ]
    )
    embeddings = {"1": np.array([1.0, 2.0], dtype=np.float32)}

    with pytest.raises(ValueError, match="Missing cached embeddings for eval_strict: missing"):
        stack_embeddings_for_rows(rows, embeddings, context="eval_strict")


def test_stack_embeddings_for_rows_can_allow_missing_pageids_explicitly() -> None:
    rows = pd.DataFrame(
        [
            {"pageid": "1", "label": "31"},
            {"pageid": "missing", "label": "22"},
        ]
    )
    embeddings = {"1": np.array([1.0, 2.0], dtype=np.float32)}

    filtered, matrix = stack_embeddings_for_rows(
        rows, embeddings, context="eval_strict", allow_missing=True
    )

    assert filtered["pageid"].tolist() == ["1"]
    assert matrix.tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError, match="No cached embeddings available for eval_strict"):
        stack_embeddings_for_rows(rows.iloc[0:0], embeddings, context="eval_strict")
