"""Embedding helpers for cached satellite image patches."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from georeset.utils.json_io import write_npz_atomic
from georeset.vision.clip_transformers import select_clip_features

UInt8ImageBatch = NDArray[np.uint8]
FloatFeatures = NDArray[np.float32]


class PatchEncoder(Protocol):
    def __call__(self, batch: UInt8ImageBatch) -> FloatFeatures: ...


def embed_patch_cache(
    *,
    patches_path: Path,
    output_path: Path,
    batch_size: int,
    encoder: PatchEncoder,
) -> None:
    data = np.load(patches_path)
    pageids = data["pageids"].astype(str)
    patches = data["patches"]
    embeddings: list[FloatFeatures] = []
    for start in range(0, len(patches), batch_size):
        encoded = encoder(patches[start : start + batch_size])
        embeddings.append(encoded.astype(np.float32))
    write_npz_atomic(output_path, pageids=pageids, embeddings=np.vstack(embeddings).astype(np.float32))


def build_transformers_clip_encoder(
    *,
    model_name: str,
    device: str,
) -> PatchEncoder:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "CLIP embedding requires the optional vision dependencies. "
            "Run with `uv run --group vision ...`."
        ) from exc

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
    model.eval()

    def encode(batch: UInt8ImageBatch) -> FloatFeatures:
        images = [Image.fromarray(patch.astype(np.uint8), mode="RGB") for patch in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = select_clip_features(model.get_image_features(**inputs), "image_embeds")
            features = features / features.norm(dim=-1, keepdim=True)
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)

    return encode
