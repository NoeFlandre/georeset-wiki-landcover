# Visualization Package

This package contains Folium map helpers for visual sanity checks.

## Module

- `map_visualizer.py`: renders CORINE polygons, Wikipedia article points, and
  OSM polygon overlays.

## Output Policy

Map HTML files are written through `write_html_map_atomic`, not direct
`folium.Map.save`, so interrupted map generation does not corrupt existing map
artifacts.

## Scope

These maps are diagnostic outputs for inspecting data quality and filtering
results. Quantitative experiment metrics live in the analysis/classification
outputs, not in this package.
