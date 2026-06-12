# Tool 3 — `divergence`

> **Locates where an eager model and its `torch.compile`d version disagree numerically, then
> attributes the cause.** When the compiled run produces different numbers than eager, the painful
> part is finding *which layer* first diverges — and then *which inductor optimization* introduced
> it. Tool 3 does both: lockstep activation capture to localize the layer, and a fusion-toggle
> causal experiment to attribute the pass (Li et al. 2026 §3.2 pain point C).

Most of compile-lens reads a `.cls.json` that the collector wrote from logs — static analysis, no
model. Tool 3 is different: localizing a divergence means running the eager and compiled models
**side by side** and comparing their intermediate activations, so **`torch` is a hard dependency**
of the capture side (the view side, below, is not). It produces a `divergences[]` section in the
`.cls.json` that the view command and the hero report read back.

---

## Part A — Theory & reference

### What it does

Two questions, in order:

1. **Where?** — Walk the model's layers in execution order and find the *first* one whose eager and
   compiled activations disagree beyond tolerance. The first one is the root site: divergence
   propagates downstream, so every later layer is just carrying the poison.
2. **Why?** — Once a layer is localized, run a causal experiment: turn candidate inductor passes off
   one at a time, recompile, and see whether the divergence disappears. The pass whose disabling
   removes it is the attributed cause.

This moves the tool from *describe* (the layer) to *attribute* (the cause).

### Localizing: lockstep activation capture

`divergence_session(eager, compiled)` registers a forward hook on **every submodule** of both
models. Each hook records that submodule's output activation, keyed by the submodule's qualified
name — so the two sides align by path (`decoder.layers.3.mlp.fc2` on one side matches the same path
on the other). After both models run, the localizer walks the captured layers in **eager execution
order** (the order the hooks fired, preserved by insertion-ordered storage) and returns the first
layer whose activations fail `torch.allclose(eager, compiled, rtol, atol)`.

A shape mismatch at a layer is itself a divergence (and the absolute difference is undefined there).
Non-tensor outputs are skipped — there is nothing to compare numerically.

### Minimal reproducer: the dynamo accuracy minifier

Once you know *which* layer diverges, the next thing you want is a *minimal* reproducer — the
smallest graph that still triggers the bug. Rather than bisect by hand, `accuracy_minifier()`
engages dynamo's built-in accuracy minifier (`torch._dynamo.config.repro_level = 4`,
`repro_after = "aot"`) for the duration of a block and restores the prior config on exit: when a
compiled model diverges from eager under it, dynamo writes a minimized repro to its repro directory.
Compose it around the run; if the dynamo repro config is absent (an older torch) it no-ops cleanly
and localization still works.

### Attributing: the fusion-toggle causal experiment

Localization tells you *where*; the causal experiment tells you *which inductor pass*. After a
divergence is confirmed, `attribute_divergence` toggles candidate inductor passes off, recompiles,
re-runs, and checks whether the divergence is gone. It minimizes (a delta-debugging-style greedy
1-minimization) to the smallest set of passes whose disabling removes the divergence — that is the
attributed cause. The candidate set is grounded in real `torch._inductor.config` flags and includes
both the default-on fusion passes and the custom post-grad pass slots where a user's own (or a buggy)
fusion pass lives.

This is a true causal step — an *intervention* (change a variable, re-run, observe), not just
another comparison. The oracle compares **outputs**, not per-layer hooks: each probe recompiles, and
`torch.compile` may trace across submodule boundaries, so an output-level check is robust where a
re-hooked comparison would not be.

> **Pass-level only.** `torch._inductor.config` exposes **pass-level** on/off switches, not per-node
> fusion control. So the finest cause Tool 3 can establish is *which pass(es), when disabled, remove
> the divergence* — never which specific fused node. The output says "pass-level" and does not
> pretend otherwise.

### What it produces

The findings serialize into a `divergences[]` array in the `.cls.json`. Each record carries the
first divergent layer (or absent when nothing diverged), the max absolute difference there (absent
for a shape mismatch), the number of layers compared, the tolerances, a human-readable suggested
cause, and — when the causal experiment ran — a nested `attribution` (the responsible passes,
whether the divergence was resolved, a summary, and how many recompile probes it took). A divergence
is an analysis result, so it is a normalized top-level array like `lint_findings`, and the
attribution is nested because it is strictly subordinate to one finding.

---

## Part B — Examples

### Localize (Python runtime API)

```python
import compile_lens as cl

with cl.divergence_session(model_eager, model_compiled) as div:
    model_eager(x)
    model_compiled(x)

findings = div.report()           # rtol/atol configurable
print(findings.first_divergent_layer, findings.max_abs_diff)
```

### Minimal reproducer

```python
from compile_lens import accuracy_minifier, divergence_session

with accuracy_minifier(), divergence_session(eager, compiled) as div:
    eager(x)
    compiled(x)
div.report()   # which layer; dynamo writes the minimized repro if it accuracy-failed
```

### Attribute the cause

```python
from compile_lens import attribute_divergence

# Run after localization shows the model diverges under torch.compile.
attribution = attribute_divergence(model_eager, (x,))
print(attribution.summary)
# e.g. "divergence removed when inductor pass(es) disabled: epilogue_fusion — pass-level attribution"
# or   "no divergence observed under torch.compile; nothing to attribute"
```

### View a stored session (no torch)

`cl divergence-view` is **view-only**: it reads the `divergences[]` already in a `.cls.json` and
renders them. The comparison ran when the artifact was written, so viewing needs no torch and no
recompilation.

```bash
cl divergence-view session.cls.json --format markdown
```

```markdown
## Divergence

### dv_001 — first divergent layer `decoder.layers.3.mlp.fc2`
- Max abs diff: 0.297
- Compared 48 layers (rtol=0.001, atol=0.00001)
- Suggested cause: divergence removed when inductor pass(es) disabled: …
- Attribution (pass-level): attributed=true, passes: …, 8 probes
```

`--format json` serializes the records. A session with no divergence reads "No divergence findings
in this session."

---

## Part C — Limitations & failure cases

### Pass-level attribution, not node-level

The causal experiment attributes to inductor **passes**, never to a specific fused node, because
`torch._inductor.config` only exposes pass-level toggles. If two passes interact, it reports a
minimal set whose disabling removes the divergence — a true, actionable statement ("disable these
and it goes away"), not a claim that one pass is solely "buggy".

### Hooks under `torch.compile`

The localizer's per-submodule hooks fire cleanly on eager models. Under `torch.compile`, the
compiled graph may trace through submodule boundaries, so per-layer hooks need not fire as expected.
The causal experiment sidesteps this by comparing final outputs rather than per-layer activations;
localization itself runs on two independently-executed models where the hooks behave normally.

### Hook overhead is small but not precisely measurable on CPU

The per-submodule hooks only `detach` (a view, no copy), so the overhead is small. Measuring it
precisely on a shared CPU is unreliable, though — the cost is smaller than the timing noise floor,
and repeated runs of the same workload swing in both directions. Tool 3 is a debug-time tool you run
when you *suspect* a divergence, not in a hot training or inference loop, so the overhead is
acceptable regardless.

### Manufacturing a divergence to test against

A *natural* inductor fusion bug is rare and hardware-specific, so the end-to-end test injects one
through inductor's public `post_grad_custom_post_pass` extension point — a real post-grad pass that
changes the numerics, producing a genuine compiled-vs-eager divergence through the real pipeline (not
a monkeypatched fake). The attribution then recompiles with that slot cleared and confirms the
divergence is attributed to it.

### `torch` is required for capture, not for viewing

Localizing and attributing run the models, so they need `torch`. `cl divergence-view` only reads a
stored `.cls.json` and needs nothing.

---

## Related

- [Tool 2a — `compile-diff`](diff.md): the *structural* graph diff, where Tool 3 is the *numerical*
  divergence localizer.
- [Tool 2b — `cache-stability`](cache_stability.md): another behavioral correctness check over the
  same `.cls.json`.
