"""Run zero-shot CLIP over cached Sentinel patch embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from georeset_wiki_landcover.experiment_paths import experiment_artifact_dir
from georeset_wiki_landcover.vision.clip_transformers import select_clip_features
from georeset_wiki_landcover.vision.clip_zero_shot import TextEncoder, run_zero_shot_evaluation


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
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(prompts: list[str]) -> NDArray[np.float32]:
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            features = select_clip_features(model.get_text_features(**inputs), "text_embeds")
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
        default=experiment_artifact_dir("clip_linear_probe_weak_labels_v1"),
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
