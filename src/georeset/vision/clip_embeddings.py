"""Embedding helpers for cached satellite image patches."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

UInt8ImageBatch = NDArray[np.uint8]
FloatFeatures = NDArray[np.float32]


class PatchEncoder(Protocol):
    def __call__(self, batch: UInt8ImageBatch) -> FloatFeatures: ...


class TorchFeatureTensor(Protocol):
    def norm(self, dim: int, keepdim: bool) -> TorchFeatureTensor: ...

    def detach(self) -> TorchFeatureTensor: ...

    def cpu(self) -> TorchFeatureTensor: ...

    def numpy(self) -> object: ...

    def __truediv__(self, other: object) -> TorchFeatureTensor: ...


def _select_image_features(output: TorchFeatureTensor | object) -> TorchFeatureTensor:
    image_embeds = getattr(output, "image_embeds", None)
    if image_embeds is not None:
        return cast(TorchFeatureTensor, image_embeds)
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return cast(TorchFeatureTensor, pooler_output)
    return output  # type: ignore[return-value]


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, pageids=pageids, embeddings=np.vstack(embeddings).astype(np.float32))


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
            features = _select_image_features(model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)

    return encode
