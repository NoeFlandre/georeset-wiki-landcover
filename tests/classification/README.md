# Classification Tests

Tests in this folder cover the article-text classification contract.

Key behaviors:

- CORINE is single-label; ambiguous multi-label outputs are auditable and not
  evaluated as valid predictions.
- OSM is multi-label; valid labels are normalized, deduplicated, and sorted.
- Shuffled controls preserve target sets while changing text assignment.
- Fingerprints include model identity, seed, temperature, labels, text source,
  task, and classification policy version.
- Metrics separate evaluated accuracy/F1 from coverage and parse failures.
