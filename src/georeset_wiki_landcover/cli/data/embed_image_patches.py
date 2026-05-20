"""Embed Experiment 014 image patch caches with a registered frozen encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from georeset_wiki_landcover.utils.json_io import write_npz_atomic
from georeset_wiki_landcover.vision.image_encoder_registry import ENCODER_REGISTRY, build_encoder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(ENCODER_REGISTRY), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    config, encoder = build_encoder(args.encoder, device=args.device)
    cache = np.load(args.patches_path)
    pageids = cache["pageids"].astype(str)
    patches = cache["patches"].astype(np.uint8)
    embeddings = []
    for start in range(0, len(patches), args.batch_size):
        embeddings.append(encoder(patches[start : start + args.batch_size]).astype(np.float32))
    output = (
        np.vstack(embeddings).astype(np.float32)
        if embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    write_npz_atomic(
        args.output_path,
        pageids=pageids,
        embeddings=output,
        encoder_name=np.asarray(args.encoder),
        model_name=np.asarray(config.model_name),
        window_m=cache["window_m"] if "window_m" in cache else np.asarray(-1, dtype=np.int64),
    )


if __name__ == "__main__":
    main()
