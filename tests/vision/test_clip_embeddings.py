from pathlib import Path

import numpy as np

from georeset.vision.clip_embeddings import embed_patch_cache


def test_embed_patch_cache_writes_normalized_embeddings_with_pageids(tmp_path: Path) -> None:
    patch_path = tmp_path / "patches.npz"
    output_path = tmp_path / "embeddings.npz"
    np.savez(
        patch_path,
        pageids=np.array(["1", "2"]),
        patches=np.array(
            [
                [[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]],
                [[[0, 2, 0], [0, 2, 0]], [[0, 2, 0], [0, 2, 0]]],
            ],
            dtype=np.uint8,
        ),
    )

    embed_patch_cache(
        patches_path=patch_path,
        output_path=output_path,
        batch_size=1,
        encoder=lambda batch: batch.reshape(len(batch), -1).mean(axis=1, keepdims=True),
    )

    output = np.load(output_path)
    assert output["pageids"].tolist() == ["1", "2"]
    assert output["embeddings"].shape == (2, 1)
    assert output["embeddings"].dtype == np.float32
