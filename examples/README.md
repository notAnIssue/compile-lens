# Examples

## Hero report

A single capture that exercises both of compile-lens's headline stories, rendered into one
self-contained HTML report.

The model ([`hero_scene.py`](./hero_scene.py)) is three `GEMM-Residual-RMSNorm-GEMM` blocks — the
shape real transformer FFNs present — followed by an affine tail `(x - bias) / scale`. A `base` and
a `head` variant differ by **exactly one thing**: the `head` flips the tail's centering subtraction
to `(bias - x) / scale`, a silent sign error that raises no exception and just produces wrong
numbers.

**Regenerate the report** from the committed artifacts (no PyTorch needed — `cl session report` is
pure Rust over the `.cls.json` files):

```bash
cl session report examples/hero_head.cls.json \
    --base examples/hero_base.cls.json --output hero.html
cl scrub hero.html        # confirms it is already share-safe (the report is born with a strict CSP)
# open hero.html
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
