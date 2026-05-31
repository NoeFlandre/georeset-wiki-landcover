# Developer Script Wrappers

This folder keeps backwards-compatible repository wrappers for developer CLIs.

## Wrapper

- `snapshot.py` wraps `georeset_wiki_landcover.cli.dev.snapshot`.
- `check_repository_hygiene.py` checks tracked files for local data, generated
  artifacts, `.env*` files, caches, and large blobs before CI or commit.

## Preferred Usage

```bash
uv run georeset-wiki-landcover-snapshot
uv run python scripts/dev/check_repository_hygiene.py
```
