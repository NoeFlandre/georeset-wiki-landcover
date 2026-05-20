# Repository Scripts

This directory contains repository-facing wrappers and Grid5000 shell launchers.

## Python Wrappers

The Python files under `scripts/data`, `scripts/analysis`, and `scripts/dev` are
thin wrappers around installable `georeset_wiki_landcover.cli.*` modules. New automation and
documentation should prefer the packaged `georeset-wiki-landcover-*` entry points.

## Cluster Scripts

The shell scripts under `scripts/cluster` sync the repository/data to Grid5000,
submit OAR GPU jobs, and perform one-shot JSON-validated syncs back to local
paths.

Auto-sync is disabled by default to avoid repeated SSH polling. Use
`SYNC_ONCE=1` for manual syncs unless you intentionally need a controlled loop.

## What Not To Put Here

Reusable Python logic should not live in top-level scripts. Put it in
`src/georeset_wiki_landcover/` and expose it through `georeset_wiki_landcover.cli` if it needs a command-line
interface.
