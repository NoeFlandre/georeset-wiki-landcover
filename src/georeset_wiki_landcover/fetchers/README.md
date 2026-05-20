# Fetchers Package

This package contains IO boundaries for external or local source data.

## Files

- `__init__.py`: marks the package; keep it free of network side effects.
- `article_summarizer.py`: LLM-backed article summarization using the shared
  Llama client.
- `data_fetcher.py`: loads local CORINE shapefiles, normalizes CRS, computes
  bounds, and saves bounds JSON.
- `landuse_evidence_summarizer.py`: LLM-backed extraction of land-use evidence,
  relevance labels, uncertainty, and mentioned land-cover concepts.
- `osm_fetcher.py`: fetches project-scoped OSM land-cover polygons from
  Overpass with retry/backoff behavior for transient HTTP failures.
- `wiki_article_type_fetcher.py`: fetches French Wikipedia category and
  page-property metadata without refetching full article content.
- `wiki_content_fetcher.py`: fetches full Wikipedia extracts by page ID with
  resumable JSON checkpoints.
- `wiki_fetcher.py`: fetches French Wikipedia geosearch metadata with
  fail-fast behavior for pipeline-critical fetches.

## Error Policy

- Pipeline-critical data fetches should fail loudly when inputs are invalid or
  remote requests cannot be trusted.
- Transient HTTP failures may use structured retry/backoff when the module owns
  the external boundary.
- Checkpoint writes should use atomic JSON helpers so resumable outputs are not
  corrupted by interruption.

## Data Contract

Fetchers read from and write to paths under `data/`, but `data/` is synced via
the Hugging Face bucket and is not committed to Git.
