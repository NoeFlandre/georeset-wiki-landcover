# CLI And Script Tests

Tests in this folder cover packaged CLI behavior plus compatibility wrappers in
top-level `scripts/`.

They verify:

- CLI defaults and environment overrides;
- experiment output shapes and manifests;
- cluster script safety constraints;
- wrapper importability without `sys.path` hacks;
- no accidental mutation of parent experiment folders.
