import pytest

from georeset_wiki_landcover.cli.csv_args import parse_csv_ints, parse_csv_strings
from georeset_wiki_landcover.cli.image_probe_args import (
    embedding_cache_paths,
    image_probe_splits_path,
    sample_weights_path,
    split_manifest_path,
    split_summary_path,
)


def test_parse_csv_strings_strips_whitespace_and_drops_empty_parts() -> None:
    assert parse_csv_strings(" clip_base, ,dinov2_base ,, clip_large ") == [
        "clip_base",
        "dinov2_base",
        "clip_large",
    ]


def test_parse_csv_ints_strips_whitespace_and_drops_empty_parts() -> None:
    assert parse_csv_ints("320, 640,,2240 ") == [320, 640, 2240]


def test_parse_csv_ints_raises_for_non_integer_parts() -> None:
    with pytest.raises(ValueError):
        parse_csv_ints("320,large")


def test_embedding_cache_paths_uses_experiment_filename_convention(tmp_path) -> None:
    assert embedding_cache_paths(
        tmp_path,
        encoders=["clip_base", "dinov2_base"],
        windows=["320", "2240"],
    ) == [
        tmp_path / "embeddings_clip_base_window_0320m.npz",
        tmp_path / "embeddings_clip_base_window_2240m.npz",
        tmp_path / "embeddings_dinov2_base_window_0320m.npz",
        tmp_path / "embeddings_dinov2_base_window_2240m.npz",
    ]


def test_image_probe_metadata_paths_use_experiment_filename_convention(tmp_path) -> None:
    assert image_probe_splits_path(tmp_path) == tmp_path / "image_probe_splits_v2.csv"
    assert sample_weights_path(tmp_path) == tmp_path / "sample_weights.csv"
    assert split_manifest_path(tmp_path) == tmp_path / "split_manifest.json"
    assert split_summary_path(tmp_path) == tmp_path / "split_summary.md"
