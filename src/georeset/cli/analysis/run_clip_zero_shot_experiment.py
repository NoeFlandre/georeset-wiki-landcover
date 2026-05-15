"""Run zero-shot CLIP over cached Sentinel patch embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from georeset.vision.clip_zero_shot import TextEncoder, run_zero_shot_evaluation


class TorchFeatureTensor(Protocol):
    def norm(self, dim: int, keepdim: bool) -> TorchFeatureTensor: ...

    def detach(self) -> TorchFeatureTensor: ...

    def cpu(self) -> TorchFeatureTensor: ...

    def numpy(self) -> object: ...

    def __truediv__(self, other: object) -> TorchFeatureTensor: ...


def _select_text_features(output: TorchFeatureTensor | object) -> TorchFeatureTensor:
    text_embeds = getattr(output, "text_embeds", None)
    if text_embeds is not None:
        return cast(TorchFeatureTensor, text_embeds)
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return cast(TorchFeatureTensor, pooler_output)
    return output  # type: ignore[return-value]


def build_transformers_clip_text_encoder(*, model_name: str, device: str) -> TextEncoder:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "CLIP zero-shot evaluation requires optional vision dependencies. "
            "Run with `uv run --group vision ...`."
        ) from exc

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
    model.eval()

    def encode(prompts: list[str]) -> NDArray[np.float32]:
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            features = _select_text_features(model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)

    return encode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, required=True)
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/clip_linear_probe_weak_labels_v1"),
    )
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_zero_shot_evaluation(
        splits_path=args.splits_path,
        embeddings_path=args.embeddings_path,
        output_dir=args.output_dir,
        text_encoder=build_transformers_clip_text_encoder(
            model_name=args.model_name,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
