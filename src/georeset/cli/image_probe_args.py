"""Shared CLI helpers for Experiment 014 image-probe artifacts."""

from __future__ import annotations

from pathlib import Path


def embedding_cache_paths(
    output_dir: Path, *, encoders: list[str], windows: list[str]
) -> list[Path]:
    """Return expected embedding cache paths for encoder/window combinations."""
    return [
        output_dir / f"embeddings_{encoder}_window_{int(window):04d}m.npz"
        for encoder in encoders
        for window in windows
    ]
