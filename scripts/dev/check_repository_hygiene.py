"""Fail if tracked repository files include local data, caches, or large blobs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024

_ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template"}
_CACHE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
}
_CACHE_NAMES = {".coverage", ".DS_Store"}
_CACHE_SUFFIXES = {".pyc", ".pyo"}


def _is_env_file(path: PurePosixPath) -> bool:
    return any(
        part not in _ENV_TEMPLATE_NAMES and (part == ".env" or part.startswith(".env."))
        for part in path.parts
    )


def _is_cache_or_metadata_path(path: PurePosixPath) -> bool:
    return (
        any(part in _CACHE_PARTS for part in path.parts)
        or path.name in _CACHE_NAMES
        or path.suffix in _CACHE_SUFFIXES
    )


def _file_size_violation(relative_path: str, repo_root: Path, max_file_bytes: int) -> str | None:
    path = repo_root / relative_path
    if not path.is_file():
        return None

    size = path.stat().st_size
    if size <= max_file_bytes:
        return None
    return f"tracked file exceeds {max_file_bytes} bytes: {relative_path} ({size} bytes)"


def check_tracked_paths(
    tracked_paths: Iterable[str],
    *,
    repo_root: Path,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[str]:
    """Return hygiene violations for tracked repository-relative paths."""

    violations: list[str] = []
    for relative_path in sorted(tracked_paths):
        path = PurePosixPath(relative_path)
        if path.parts and path.parts[0] == "data":
            violations.append(f"tracked private/local path is forbidden: {relative_path}")
        if path.parts and path.parts[0] == "build":
            violations.append(f"tracked generated artifact path is forbidden: {relative_path}")
        if _is_env_file(path):
            violations.append(f"tracked environment file is forbidden: {relative_path}")
        if _is_cache_or_metadata_path(path):
            violations.append(f"tracked cache/metadata path is forbidden: {relative_path}")

        size_violation = _file_size_violation(relative_path, repo_root, max_file_bytes)
        if size_violation is not None:
            violations.append(size_violation)
    return violations


def _tracked_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check tracked files for accidental local data, generated artifacts, "
            "environment files, caches, and large blobs."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to inspect. Defaults to the current directory.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Maximum allowed size for a tracked file before failing.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    tracked_paths = _tracked_paths(repo_root)
    violations = check_tracked_paths(
        tracked_paths,
        repo_root=repo_root,
        max_file_bytes=args.max_file_bytes,
    )
    if violations:
        print("ERROR: repository hygiene check failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1

    print(f"Repository hygiene check passed: {len(tracked_paths)} tracked files inspected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
