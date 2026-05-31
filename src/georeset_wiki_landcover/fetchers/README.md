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

## Guardrails

- CORINE polygon loading fails before downstream sampling when the input file is
  missing or does not expose the required `code_18` column. This prevents later
  stages from sampling from an unknown or incompatible land-cover schema.
- Loaded CORINE data is treated as an explicit cache: callers that reach bounds
  or sampling code without a populated dataset get `RuntimeError: Dataset could
  not be loaded` instead of relying on assertions that can be disabled.
- CORINE sampling fails when filters remove all rows or when the requested
  sample size exceeds the available filtered polygons. This prevents silent
  pilot/full-output confusion caused by undersized samples.
- Bounds writes accept normal filesystem path objects and go through the shared
  atomic JSON writer, keeping output-file behavior consistent with the rest of
  the pipeline.
- Wikipedia article-type metadata fetching rejects non-positive retry budgets
  with `ValueError: max_attempts must be at least 1`. Retry exhaustion uses a
  concrete exception path instead of an assertion fallback, so optimized Python
  runs still fail clearly.
