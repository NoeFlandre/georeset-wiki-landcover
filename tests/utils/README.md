# Utility Tests

Tests in this folder cover cross-cutting helpers.

The main focus is atomic file IO:

- JSON/text/CSV writes use temp files plus `os.replace`;
- GeoJSON, Folium HTML maps, and parquet outputs are written atomically;
- existing files survive serialization, rendering, parquet, or replace
  failures.
