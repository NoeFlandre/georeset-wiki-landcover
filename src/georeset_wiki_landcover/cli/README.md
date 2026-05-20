# CLI Package

This package contains installable command-line entry points exposed through
`[project.scripts]` in `pyproject.toml`.

Prefer these `georeset-wiki-landcover-*` commands over importing top-level `scripts.*`
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
- Reusable logic belongs in package modules such as `georeset_wiki_landcover.classification`,
  `georeset_wiki_landcover.fetchers`, `georeset_wiki_landcover.spatial`, or `georeset_wiki_landcover.analysis`.
- File outputs should use atomic helpers from `georeset_wiki_landcover.utils.json_io`.
- Top-level `scripts/` modules exist only as repository compatibility wrappers.

## Common Commands

```bash
uv run georeset-wiki-landcover-snapshot
uv run georeset-wiki-landcover-filter-pipeline --dry-run
uv run georeset-wiki-landcover-classify-articles --help
uv run georeset-wiki-landcover-summarize-classification-experiment --help
uv run georeset-wiki-landcover-compute-corine-spatial-confidence --help
uv run georeset-wiki-landcover-evaluate-spatial-confidence --help
```
