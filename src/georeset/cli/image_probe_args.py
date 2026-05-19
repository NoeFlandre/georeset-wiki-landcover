"""Shared CLI helpers for Experiment 014 image-probe artifacts."""

from __future__ import annotations

from pathlib import Path


def image_probe_splits_path(output_dir: Path) -> Path:
    """Return the canonical Experiment 014 split metadata path."""
    return output_dir / "image_probe_splits_v2.csv"


def sample_weights_path(output_dir: Path) -> Path:
    """Return the canonical Experiment 014 sample-weight metadata path."""
    return output_dir / "sample_weights.csv"


def split_manifest_path(output_dir: Path) -> Path:
    """Return the canonical Experiment 014 split manifest path."""
    return output_dir / "split_manifest.json"


def split_summary_path(output_dir: Path) -> Path:
    """Return the canonical Experiment 014 split summary path."""
    return output_dir / "split_summary.md"


def embedding_cache_paths(
    output_dir: Path, *, encoders: list[str], windows: list[str]
) -> list[Path]:
    """Return expected embedding cache paths for encoder/window combinations."""
    return [
        output_dir / f"embeddings_{encoder}_window_{int(window):04d}m.npz"
        for encoder in encoders
        for window in windows
    ]
