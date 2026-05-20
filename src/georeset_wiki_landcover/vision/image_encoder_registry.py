"""Registry for image encoders used by Experiment 014."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

UInt8ImageBatch = NDArray[np.uint8]
FloatFeatures = NDArray[np.float32]


class PatchEncoder(Protocol):
    def __call__(self, batch: UInt8ImageBatch) -> FloatFeatures: ...


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    model_name: str
    builder: Callable[[str, str], PatchEncoder]


def _normalize(features: object) -> FloatFeatures:
    array = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(array / norms, dtype=np.float32)


def _encoder_output_to_numpy(output: object) -> FloatFeatures:
    tensor: Any = output
    if not hasattr(tensor, "detach"):
        tensor = getattr(output, "image_embeds", None)
    if tensor is None:
        tensor = getattr(output, "pooler_output", None)
    if tensor is None:
        hidden_state = getattr(output, "last_hidden_state", None)
        if hidden_state is not None:
            tensor = hidden_state[:, 0]
    if tensor is None or not hasattr(tensor, "detach"):
        raise TypeError(f"Unsupported encoder output type: {type(output).__name__}")
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def build_clip_encoder(model_name: str, device: str) -> PatchEncoder:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("Image encoding requires `uv run --group vision ...`.") from exc

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
    model.eval()

    def encode(batch: UInt8ImageBatch) -> FloatFeatures:
        images = [Image.fromarray(patch.astype(np.uint8), mode="RGB") for patch in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return _normalize(_encoder_output_to_numpy(features))

    return encode


def build_dinov2_encoder(model_name: str, device: str) -> PatchEncoder:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("Image encoding requires `uv run --group vision ...`.") from exc

    processor = AutoImageProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(batch: UInt8ImageBatch) -> FloatFeatures:
        images = [Image.fromarray(patch.astype(np.uint8), mode="RGB") for patch in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = outputs.last_hidden_state[:, 0]
        return _normalize(_encoder_output_to_numpy(features))

    return encode


ENCODER_REGISTRY: dict[str, EncoderConfig] = {
    "clip_base": EncoderConfig(
        name="clip_base",
        model_name="openai/clip-vit-base-patch32",
        builder=build_clip_encoder,
    ),
    "clip_large": EncoderConfig(
        name="clip_large",
        model_name="openai/clip-vit-large-patch14",
        builder=build_clip_encoder,
    ),
    "dinov2_base": EncoderConfig(
        name="dinov2_base",
        model_name="facebook/dinov2-base",
        builder=build_dinov2_encoder,
    ),
}


def build_encoder(name: str, *, device: str) -> tuple[EncoderConfig, PatchEncoder]:
    config = ENCODER_REGISTRY[name]
    return config, config.builder(config.model_name, device)
