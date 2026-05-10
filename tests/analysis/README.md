# Analysis Tests

Tests in this folder cover derived experiment outputs, especially spatial
confidence reevaluation.

They verify that:

- spatial subsets select the expected page IDs;
- `all_available_spatial_confidence` excludes predictions without spatial rows;
- majority baselines and shuffled deltas are recomputed per subset;
- parent experiment directories are not mutated;
- manifests record parent/spatial experiment IDs.
