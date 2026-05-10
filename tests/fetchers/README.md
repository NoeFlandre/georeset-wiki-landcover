# Fetcher Tests

Tests in this folder cover external data boundaries and local source-data
loading.

They focus on:

- CRS handling and required columns for CORINE loading;
- OSM bounds validation and retry behavior;
- Wikipedia metadata/content failure behavior;
- resumable content checkpointing with normalized page IDs;
- summary generation without leaking private model thinking fields.
