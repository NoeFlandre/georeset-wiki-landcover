"""Embed cached satellite patches with a frozen CLIP image encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from georeset.vision.clip_embeddings import build_transformers_clip_encoder, embed_patch_cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    encoder = build_transformers_clip_encoder(model_name=args.model_name, device=args.device)
    embed_patch_cache(
        patches_path=args.patches_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        encoder=encoder,
    )


if __name__ == "__main__":
    main()
