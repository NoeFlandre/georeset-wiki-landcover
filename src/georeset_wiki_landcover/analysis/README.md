# Analysis Package

Reusable, non-LLM helpers for experiment analysis, metric computation, metadata
loading, and quality/spatial subset construction. Command-line parsing and
artifact writing live in `georeset_wiki_landcover.cli.analysis`.

## Files

- `__init__.py`: marks the package; keep it free of heavy imports.
- `article_type_classifier.py`: deterministic French Wikipedia category-to-type
  classifier used for article-type diagnostics.
- `article_type_metadata_loading.py`: loads article-type metadata artifacts and
  joins them to experiment frames.
- `corine_polygon_stats.py`: computes area-weighted CORINE class distributions
  inside OSM polygons with CRS-aware area handling.
- `distribution_summary.py`: summarizes distribution tables into compact
  tabular diagnostics.
- `evaluation_metrics.py`: shared single-label and multi-label metric helpers
  used by analysis CLIs.
- `evidence_metadata_loading.py`: loads land-use evidence metadata and exposes
  normalized relevance/uncertainty fields.
- `label_universe.py`: builds shared label universes so metrics compare the
  same class set across models and subsets.
- `list_normalization.py`: small normalization helpers for serialized list-like
  values in CSV/Markdown-derived artifacts.
- `pageid_frames.py`: helpers for tabular artifacts keyed by Wikipedia page ID.
- `prediction_loading.py`: loads frozen classification prediction records into
  analysis-ready frames.
- `quality_subsets.py`: defines reusable relevance, quality, and 250 m spatial
  subset masks.
- `shuffled_deltas.py`: computes aligned-vs-shuffled deltas for text-source
  controls.
- `spatial_confidence_loading.py`: loads CORINE spatial-confidence artifacts and
  normalizes spatial-confidence columns.

## Design Boundaries

- Keep reusable, mostly pure analysis logic here.
- Keep command-line parsing in `georeset_wiki_landcover.cli.analysis`.
- Keep LLM calls out of this package; analysis should consume frozen artifacts.
- Write files through atomic helpers from `georeset_wiki_landcover.utils.json_io` in the CLI
  orchestration layer.
