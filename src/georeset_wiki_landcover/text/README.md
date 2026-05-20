# Text Artifacts

This package builds deterministic text sources used by classification experiments.

The modules keep artifact construction separate from CLI parsing and file IO.
Shared helpers live in small neutral modules such as `labels`, `record_access`,
and `title_scrubbing` so artifact builders do not depend on each other's private
functions.

## Files

- `__init__.py`: marks the package; keep it free of artifact-building side
  effects.
- `evidence_cards.py`: builds compact structured evidence-card text records from
  existing metadata.
- `evidence_highlights.py`: prepends matched evidence sentences to full article
  text for highlighted-content experiments.
- `labels.py`: shared display labels for deterministic text artifacts.
- `record_access.py`: small access helpers for JSON records used by text
  artifact builders.
- `retrieved_evidence_windows.py`: builds raw neighboring-sentence windows
  around evidence matches plus random-window controls.
- `title_scrubbing.py`: removes article-title leakage from generated text
  artifacts.

Changes here should preserve generated record schemas and text outputs unless an
experiment explicitly calls for a new artifact version.
