# Examples

## Hero report

A single capture that exercises both of compile-lens's headline stories, rendered into one
self-contained HTML report.

The model ([`hero_scene.py`](./hero_scene.py)) is three `GEMM-Residual-RMSNorm-GEMM` blocks — the
shape real transformer FFNs present — followed by an affine tail `(x - bias) / scale`. A `base` and
a `head` variant differ by **exactly one thing**: the `head` flips the tail's centering subtraction
to `(bias - x) / scale`, a silent sign error that raises no exception and just produces wrong
numbers.

**Just open [`hero.html`](./hero.html).** The rendered report is committed and CI-checked against
the current renderer, so there is nothing to build or install — open it in a browser.

**Regenerate it** (only needed after a renderer change) with one command, which builds the current
`cl` binary and re-renders so the output can never come from a stale binary:

```bash
./scripts/render_hero.sh        # writes examples/hero.html
```

What the report shows:

- **Pillar — regression governance (IR Diff section).** `cl diff` recovers the sign-flip as exactly
  one `modified` node — the `sub` — at **100% match coverage and 100% anchor uniqueness**. The
  pitch: "this PR changed exactly one operation, here it is, and we are confident."
- **Crown jewel — fusion opportunities (CODA section).** Each block is a `Pattern A` Inductor leaves
  unfused; the report flags all three with an analytical HBM-traffic estimate — a **memory-bound
  upper bound for ranking, not a measured speedup** (real is ~1.05–1.15× on LLaMA shapes, GEMMs
  being compute-bound).

**Re-capture from scratch** (needs PyTorch; CPU is fine):

```bash
python examples/hero_scene.py     # rewrites hero_base.cls.json / hero_head.cls.json
```

The committed artifacts are pinned for determinism and scrubbed to `public-safe`, so they carry no
host or command — safe to share as-is.

> **On `cl` discovery.** The script and the CLI use the binary built into this repo
> (`target/release/cl`); the in-process hero loop `cl.session().report().save_html()` instead shells
> out to a `cl` on your `PATH` (or `$CL_BIN`). Shipping the Rust binary alongside the Python wheel so
> that always resolves is part of the release milestone, not yet wired — for now, set
> `CL_BIN=$(pwd)/target/release/cl` if you drive the hero loop from Python.
