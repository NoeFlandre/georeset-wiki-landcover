# Data Script Wrappers

This folder keeps backwards-compatible repository wrappers for packaged data
CLIs.

## Wrappers

- `filter_pipeline.py` wraps `georeset.cli.data.filter_pipeline`.
- `summarize_articles.py` wraps `georeset.cli.data.summarize_articles`.
- `summarize_landuse_evidence.py` wraps
  `georeset.cli.data.summarize_landuse_evidence`.
- `classify_articles.py` wraps `georeset.cli.data.classify_articles`.
- `compute_corine_spatial_confidence.py` wraps
  `georeset.cli.data.compute_corine_spatial_confidence`.

## Preferred Usage

Use packaged commands in new docs and automation:

```bash
uv run georeset-filter-pipeline --dry-run
uv run georeset-summarize-articles --help
uv run georeset-classify-articles --help
uv run georeset-compute-corine-spatial-confidence --help
```
