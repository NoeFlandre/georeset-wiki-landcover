# Test Suite

The tests document the project contracts as much as they verify behavior.

## Test Areas

- `tests/classification`: label policies, prediction parsing, runner behavior,
  resumability, and metrics.
- `tests/fetchers`: CORINE, OSM, Wikipedia metadata/content, and summarization
  boundaries.
- `tests/spatial`: CORINE spatial-confidence geometry and entropy behavior.
- `tests/analysis`: spatial-subset reevaluation and experiment artifact
  generation.
- `tests/scripts`: packaged CLI behavior and repository wrapper compatibility.
- `tests/utils`: atomic file writer contracts.
- `tests/visualization`: map generation safety and empty-data behavior.

## Quality Gate

```bash
uv lock --check
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/dev/check_repository_hygiene.py
PYTHONDONTWRITEBYTECODE=1 uv run georeset-wiki-landcover-classify-articles --help
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/reproduce_small.py \
  --output-dir build/reproducibility/small \
  --clean
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
PYTHONDONTWRITEBYTECODE=1 uv run ruff check .
PYTHONDONTWRITEBYTECODE=1 uv run ruff format --check .
PYTHONDONTWRITEBYTECODE=1 uv run mypy src scripts
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q
```

The default pytest configuration measures `georeset_wiki_landcover.classification` coverage
with a 95% fail-under threshold.
