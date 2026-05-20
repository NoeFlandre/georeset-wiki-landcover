from pathlib import Path

from georeset_wiki_landcover.experiment_paths import (
    experiment_artifact_dir,
    experiment_artifact_file,
    resolve_existing_experiment_artifact_dir,
)


def test_experiment_artifact_dir_uses_numbered_docs_style_group() -> None:
    assert experiment_artifact_dir("clip_linear_probe_weak_labels_v1") == Path(
        "data/experiments/012_clip_linear_probe_weak_labels/clip_linear_probe_weak_labels_v1"
    )
    assert experiment_artifact_dir("quality_weighted_multiscale_image_probe_v1") == Path(
        "data/experiments/014_quality_weighted_multiscale_image_probe/"
        "quality_weighted_multiscale_image_probe_v1"
    )


def test_experiment_artifact_file_appends_child_path() -> None:
    assert experiment_artifact_file(
        "corine_spatial_confidence_v1", "spatial_confidence.csv"
    ) == Path(
        "data/experiments/002_corine_spatial_confidence/"
        "corine_spatial_confidence_v1/spatial_confidence.csv"
    )


def test_resolve_existing_experiment_artifact_dir_prefers_grouped_path(tmp_path: Path) -> None:
    grouped = tmp_path / "012_clip_linear_probe_weak_labels" / "clip_linear_probe_weak_labels_v1"
    legacy = tmp_path / "clip_linear_probe_weak_labels_v1"
    grouped.mkdir(parents=True)
    legacy.mkdir()

    assert (
        resolve_existing_experiment_artifact_dir("clip_linear_probe_weak_labels_v1", root=tmp_path)
        == grouped
    )


def test_resolve_existing_experiment_artifact_dir_falls_back_to_legacy_flat_path(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "clip_linear_probe_weak_labels_v1"
    legacy.mkdir()

    assert (
        resolve_existing_experiment_artifact_dir("clip_linear_probe_weak_labels_v1", root=tmp_path)
        == legacy
    )
