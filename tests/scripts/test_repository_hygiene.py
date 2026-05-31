from pathlib import Path

from scripts.dev.check_repository_hygiene import check_tracked_paths


def test_repository_hygiene_reports_forbidden_tracked_paths(tmp_path: Path) -> None:
    large_file = tmp_path / "docs/large.bin"
    large_file.parent.mkdir(parents=True)
    large_file.write_bytes(b"01234567890")

    violations = check_tracked_paths(
        [
            "data/raw/input.geojson",
            "build/reproducibility/small/manifest.json",
            ".env",
            "config/.env.local",
            "scripts/__pycache__/module.pyc",
            "docs/large.bin",
        ],
        repo_root=tmp_path,
        max_file_bytes=10,
    )

    assert "tracked private/local path is forbidden: data/raw/input.geojson" in violations
    assert (
        "tracked generated artifact path is forbidden: build/reproducibility/small/manifest.json"
    ) in violations
    assert "tracked environment file is forbidden: .env" in violations
    assert "tracked environment file is forbidden: config/.env.local" in violations
    assert "tracked cache/metadata path is forbidden: scripts/__pycache__/module.pyc" in violations
    assert "tracked file exceeds 10 bytes: docs/large.bin (11 bytes)" in violations


def test_repository_hygiene_allows_small_source_files(tmp_path: Path) -> None:
    source_file = tmp_path / "src/georeset_wiki_landcover/example.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("x = 1\n", encoding="utf-8")

    violations = check_tracked_paths(
        ["src/georeset_wiki_landcover/example.py"],
        repo_root=tmp_path,
        max_file_bytes=10,
    )

    assert violations == []
