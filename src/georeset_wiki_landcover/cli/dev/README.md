# Developer CLIs

This package contains lightweight developer diagnostics that are useful during
local exploration but are not part of the research experiment protocol.

## Commands

- `__init__.py`: marks the command package; keep it side-effect free.
- `snapshot.py`: prints a quick snapshot of the CORINE dataset, including
  columns, CRS, shape, bounds, class counts, and sample rows.

## Design Boundary

Keep developer-only diagnostics here. If a diagnostic becomes part of a
repeatable experiment or produces artifacts, move the reusable logic into a
package module and expose it through `georeset_wiki_landcover.cli.data` or
`georeset_wiki_landcover.cli.analysis`.
