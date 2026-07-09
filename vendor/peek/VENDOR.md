# Vendored copy of peek-ai

This directory contains a vendored copy of `peek-ai`
([zhuohangu/peek](https://github.com/zhuohangu/peek)) at commit
`57de91ac` (2026-05-20). License: Apache-2.0 (see `LICENSE`).

The package imports as `peek` exactly as the upstream PyPI/Git install
would, so `from peek import CachePolicy` etc. is unchanged.

## Why vendored

We patch peek-ai internals (Distiller, Cartographer, Evictor) to
experiment with improvements identified in our benchmark on
`oolongbench/oolong-synth`. Vendoring lets us iterate freely while
keeping a clean upstream reference for diffing.

## Patches

Each patch is documented inline at the change site with a `# peek-patch:`
comment and corresponds to a hypothesis in
`docs/peek-bench/PEEK-EXPERIMENTS.md` (Phase 4).

When the patch list stabilises and improves results meaningfully, the
intent is to contribute them back upstream.

## Refreshing from upstream

To re-vendor from a newer upstream commit, replace the files under this
directory with the new content and re-apply the `# peek-patch:` markers.
Update the commit ref above.
