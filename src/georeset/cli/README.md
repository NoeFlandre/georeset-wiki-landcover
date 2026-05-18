# CLI Package

This package contains installable command-line entry points exposed through
`[project.scripts]` in `pyproject.toml`.

Prefer these `georeset-*` commands over importing top-level `scripts.*`
wrappers in new documentation or automation.

## Files And Subpackages

- `__init__.py`: marks the CLI package and should not import command modules.
- `analysis/`: commands that read frozen artifacts and produce derived metrics,
  summaries, or comparisons.
- `data/`: commands that create or update data artifacts used by experiments.
- `dev/`: local developer diagnostics that are not part of the formal research
  protocol.

## Design Rules

- CLIs parse arguments, configure paths, call reusable package modules, and
  write artifacts.
- Reusable logic belongs in package modules such as `georeset.classification`,
  `georeset.fetchers`, `georeset.spatial`, or `georeset.analysis`.
- File outputs should use atomic helpers from `georeset.utils.json_io`.
- Top-level `scripts/` modules exist only as repository compatibility wrappers.

## Common Commands

```bash
uv run georeset-snapshot
uv run georeset-filter-pipeline --dry-run
uv run georeset-classify-articles --help
uv run georeset-summarize-classification-experiment --help
uv run georeset-compute-corine-spatial-confidence --help
uv run georeset-evaluate-spatial-confidence --help
```
