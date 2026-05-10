# Utilities Package

This package contains small cross-cutting helpers.

## Modules

- `json_io.py`: atomic file writers for JSON, text, CSV, GeoJSON, Folium HTML
  maps, and optional parquet outputs.

## Atomic Write Policy

Pipeline artifacts should be written through same-directory temporary files and
`os.replace`. This protects resumable outputs from interruption and avoids
leaving partially written JSON, CSV, GeoJSON, map HTML, or parquet files at the
final path.

Direct production writes such as `open(..., "w")`, `json.dump`, `.to_csv`,
`.to_file`, `.save`, and `.to_parquet` should stay confined to the atomic helper
module or tests.
