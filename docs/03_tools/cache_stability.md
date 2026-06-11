# Tool 2b — `cache-stability`

> **Catches a silently-wrong `torch.compile` cache bug.** Across a run's iterations, the model's
> internal state drifts, the compiled graph is reused from cache anyway, and the output stays
> *frozen* — so the numerics are stale while the run looks fine. The user can't see it: nothing
> errors, nothing crashes (Li et al. 2026 §3.2.1, Listing 2).

A compiled graph is keyed by some cache key. If a model has mutable internal state (a buffer, a
scalar attribute) and that state changes in a way the key doesn't capture, `torch.compile` serves a
**stale** compiled graph: the computation uses old state and the output is wrong, silently. This is
the behavioral analogue of Tool 2a's operand-order false-negative — the dangerous bugs are the ones
that don't announce themselves.

`cache-stability` reads the **per-iteration** data the collector already captures (`iterations[]` —
no new capture) and flags the signature of this bug.

---

## Part A — Theory & reference

### Mode B — single-run anomaly

Given one run's `iterations[]`, for each iteration after the first, the signature is **three
conditions at once**:

```
state_mutated  = iterations[i].internal_state_snapshot.module_attrs_changed is non-empty
cache_reused   = iterations[i].cache_hit is true
output_frozen  = iterations[i].output_signature == iterations[i-1].output_signature   (both known)

if state_mutated and cache_reused and output_frozen:
    -> high-severity finding: graph_caching_mutable_state_not_invalidated
```

Only the **conjunction** is suspicious — each condition alone is normal. State drifts between
iterations all the time (a step counter, a running statistic). A cache hit is the optimization
working. A frozen output is a steady state. But all three together is the contradiction: the state
changed, the cache *didn't* invalidate, and the output *didn't* move — the cache key missed the
state change and served a stale graph.

`output_frozen` requires a **known** equality — if either signature is absent, the output is
*unknown*, not frozen, and nothing is flagged. (Absent data is never treated as evidence.)

### Mode A — diff-based regression (rides on `cl diff`)

Given a `base` and a `head` session (a change's before/after), Mode A reports a **regression the
change introduced**, not a standalone anomaly:

- **output instability introduced** — the base run's output was stable across all its iterations,
  but the head's output changed at some iteration. Attribution requires the base to have been
  steady; if the base was already unstable, head instability is not the change's fault.
- **new recompilations** — iterations where the head triggered a recompilation the base did not (by
  iteration index); a recompilation both sides share is not a regression.

Mode A rides on `cl diff` because it needs a base and a head — exactly `cl diff`'s input.

### The data it reads

Both modes read `iterations[]` (the collector's multi-iteration capture): each entry's `cache_hit`,
`output_signature` (a hash over the run's output tensors), `recompilation_triggered`, and
`internal_state_snapshot.module_attrs_changed` (which mutable attributes drifted since the previous
iteration). No graph structure is needed — this is a *behavioral* check, distinct from Tool 2a's
*structural* graph diff.

---

## Part B — Examples

### Producing the inputs

Capture multiple iterations with the collector so there is per-iteration data to analyze:

```python
from compile_lens.collectors.artifact import CompileArtifactCollector

c = CompileArtifactCollector("session.cls.json")
c.capture(model, example_input, iterations=8)   # run 8 times, record per-iteration state
c.finalize()
```

### Mode B — `cl cache-stability`

```bash
cl cache-stability session.cls.json --format markdown
```

```markdown
## Cache Stability

2 high-severity finding(s).

- **iteration 1** (high) — `graph_caching_mutable_state_not_invalidated`; changed: step_counter
- **iteration 2** (high) — `graph_caching_mutable_state_not_invalidated`; changed: step_counter
```

`--format json` serializes the findings. A clean run reads "No cache-stability anomalies detected."

### Mode A — the cache-stability section of `cl diff`

```bash
cl diff --base before.cls.json --head after.cls.json --format markdown
```

Below the graph diff, `cl diff` appends:

```markdown
## Cache Stability (diff)

No cache-stability regression.
```

— or, when a regression is found, the iteration where instability was introduced and any new
recompilations. `--format json` emits `{ "graph_diff": …, "cache_stability": … }`.

---

## Part C — Limitations & failure cases

### Needs multi-iteration capture

Both modes read `iterations[]`. A session captured with a single iteration (or none) has nothing to
compare across, so the check is silent — not because the model is clean, but because there's no
per-iteration data. Capture with `iterations=N` (N ≥ 2) to exercise it.

### Precision over recall — the conjunction is deliberate

The three-condition conjunction is specific on purpose. A cache-stability tool that false-positives
on normal stateful modules gets disabled, after which it catches nothing — so a false positive costs
more than a missed case. The conjunction (state mutated *and* cache reused *and* output frozen) keeps
the false-positive rate low; a normal stateful module, where the cache correctly invalidates and the
output moves, stays silent. The committed false-positive baseline fixture guards this.

### Absent data is not evidence

`output_frozen` (Mode B) and the output-change check (Mode A) require *known* signatures. A missing
`output_signature` is unknown, not "unchanged" — so it is never flagged. The cost is a possible
missed case when data is incomplete, which is the right trade for a correctness tool: don't accuse on
absent evidence.

### Mode A attributes to the base

Mode A only calls head instability a regression if the base was a steady baseline. A base that
already varies its output makes head variation un-attributable to the change — so it reports none.

---

## Related

- [Tool 2a — `compile-diff`](diff.md): the structural graph diff `cl diff` carries alongside this.
- [Tool 1 — `recompile-summary`](recompile_summary.md): the other analyzer over the same `.cls.json`.
