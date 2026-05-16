# Text Tests

Tests in this folder cover deterministic text artifact construction.

They assert generated strings, metadata fields, JSON-safe values, and title
scrubbing behavior for the text sources used by classification experiments.
Helper-module tests cover shared label rendering, record access, and scalar
normalization.

When refactoring text artifact modules, keep these tests focused on preserving
existing outputs and schemas.
