"""CLI for validating project reproducibility artifact directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from georeset_wiki_landcover.reproducibility.artifact_validator import (
    validate_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("build/reproducibility/small"),
        help="Artifact root to validate.",
    )
    parser.add_argument("--profile", choices=["small", "full"], default="small")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    violations = validate_artifacts(args.root, profile=args.profile)
    if violations:
        for violation in violations:
            print(f"ERROR: {violation}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Artifact validation passed: {args.root} ({args.profile})")


if __name__ == "__main__":
    main()
