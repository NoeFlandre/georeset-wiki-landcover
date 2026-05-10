# Analysis Package

This package contains reusable, non-LLM analysis helpers for geospatial overlap
and distribution summaries.

## Modules

- `corine_polygon_stats.py`: computes area-weighted CORINE class distributions
  inside OSM polygons. It expects geometries with valid CRS metadata and uses
  projected-area calculations where needed.
- `distribution_summary.py`: summarizes distribution tables into compact
  tabular diagnostics.

## Design Boundaries

- Keep pure analysis logic here.
- Keep command-line parsing in `georeset.cli.analysis`.
- Keep file writing through atomic helpers from `georeset.utils.json_io` when
  analysis code is used by an orchestration layer.

## Typical Flow

1. Load CORINE and OSM polygons in a CLI or fetcher.
2. Filter/project geometries as needed.
3. Call analysis helpers to compute area/share tables.
4. Write CSV/Markdown outputs from the CLI layer.
