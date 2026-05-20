# Data Script Wrappers

This folder keeps backwards-compatible repository wrappers for packaged data
CLIs.

## Wrappers

- `filter_pipeline.py` wraps `georeset_wiki_landcover.cli.data.filter_pipeline`.
- `summarize_articles.py` wraps `georeset_wiki_landcover.cli.data.summarize_articles`.
- `summarize_landuse_evidence.py` wraps
  `georeset_wiki_landcover.cli.data.summarize_landuse_evidence`.
- `classify_articles.py` wraps `georeset_wiki_landcover.cli.data.classify_articles`.
- `compute_corine_spatial_confidence.py` wraps
  `georeset_wiki_landcover.cli.data.compute_corine_spatial_confidence`.

## Preferred Usage

Use packaged commands in new docs and automation:

```bash
uv run georeset-wiki-landcover-filter-pipeline --dry-run
uv run georeset-wiki-landcover-summarize-articles --help
uv run georeset-wiki-landcover-classify-articles --help
uv run georeset-wiki-landcover-compute-corine-spatial-confidence --help
```
