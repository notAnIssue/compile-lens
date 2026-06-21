# Examples

## Hero report

A single `cl.capture()` call over one `torch.compile` workload, rendered into one self-contained HTML
report in which every CPU-capturable tool fires.

The model ([`hero_scene.py`](./hero_scene.py)) is three `GEMM-Residual-RMSNorm-GEMM` blocks — the
shape real transformer FFNs present — followed by an affine tail. It has several real problems at
once, the way a messy PR does: the `head` variant flips the tail's centering subtraction to
`(bias - x) / scale` (a silent sign regression); a custom post-grad "fusion" pass rewrites an `add`
into a `sub` (a miscompile that makes the compiled model diverge from eager); and a helper writes
in-place through a `view` that aliases its input (a correctness risk). Run across four sequence
lengths, it also recompiles. `cl.capture()` drives all of that in one call.

**Just open [`hero.html`](./hero.html).** The rendered report is committed and CI-checked against
the current renderer, so there is nothing to build or install — open it in a browser.

**Regenerate it** (only needed after a renderer change) with one command, which builds the current
`cl` binary and re-renders so the output can never come from a stale binary:

```bash
./scripts/render_hero.sh        # writes examples/hero.html
```

What the report shows — one capture, the whole cross-tool triage, **each finding located in source**:

- **Recompile (Tool 1).** Three recompiles attributed to the changing batch axis, with the
  `torch._dynamo.mark_dynamic` fix.
- **IR Diff (Tool 2a).** The sign-flip recovered as exactly one `modified` node at **100% match
  coverage** — "this PR changed exactly one operation, here it is."
- **Cache stability (Tool 2b).** The repeated-shape iterations checked clean — the honest no-bug
  case, rendered as a check rather than left empty.
- **Divergence (Tool 3).** The first layer where the miscompiled model disagrees with eager, with
  the max absolute difference.
- **Lint (Tool 4).** The in-place-on-alias candidate, located in the source.
- **Fusion (Tool 6).** Each block is a `Pattern A` Inductor leaves unfused; an analytical
  HBM-traffic estimate ranks all three — a **memory-bound upper bound, not a measured speedup**
  (real is ~1.05–1.15× on LLaMA shapes, GEMMs being compute-bound).
- **Roofline (Tool 5)** needs a measured GPU kernel profile and says so — it is not capturable on CPU.

**Re-capture from scratch** (needs PyTorch; CPU is fine):

```bash
python examples/hero_scene.py     # rewrites hero_base.cls.json / hero_head.cls.json
```

The committed artifacts are pinned for determinism and carry no host or command (default-strict
redaction, source paths normalized to `{repo}/…`) — safe to share as-is.

> **On `cl` discovery.** The script and the CLI use the binary built into this repo
> (`target/release/cl`); the in-process report loop `cl.capture(...).report().save_html()` instead
> shells out to a `cl` on your `PATH` (or `$CL_BIN`). Shipping the Rust binary alongside the Python wheel so
> that always resolves is part of the release milestone, not yet wired — for now, set
> `CL_BIN=$(pwd)/target/release/cl` if you drive the hero loop from Python.
