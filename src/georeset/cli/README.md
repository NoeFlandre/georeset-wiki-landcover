# CLI Package

This package contains installable command-line entry points exposed through
`[project.scripts]` in `pyproject.toml`.

Prefer these `georeset-*` commands over importing top-level `scripts.*`
wrappers in new documentation or automation.

## Entry Point Groups

- `georeset.cli.data`: data preparation, article summarization, classification,
  and spatial-confidence generation.
- `georeset.cli.analysis`: experiment summarization and reevaluation commands.
- `georeset.cli.dev`: small developer diagnostics.

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
