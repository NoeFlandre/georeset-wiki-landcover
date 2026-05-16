# Text Artifacts

This package builds deterministic text sources used by classification experiments.

The modules keep artifact construction separate from CLI parsing and file IO.
Shared helpers live in small neutral modules such as `labels`, `record_access`,
and `title_scrubbing` so artifact builders do not depend on each other's private
functions.

Changes here should preserve generated record schemas and text outputs unless an
experiment explicitly calls for a new artifact version.
