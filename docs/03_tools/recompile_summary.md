# Tool 1 — `recompile-summary`

> **Attributes a `torch.compile` session's recompilations to their root cause.** It reads a
> collected `.cls.json`, clusters the guard failures by the **dynamic axis** that drove each
> recompile (e.g. *dim 0 of `x` keeps changing size*), and emits a ranked, axis-precise fix
> (`torch._dynamo.mark_dynamic(x, 0)`) — not just a count.

Recompilation is one of the most common reasons a `torch.compile`d model is slow: every time a
guard fails, Dynamo throws away the compiled graph and re-traces. `recompile-summary` answers
*how many times, why, on which tensor/axis, and what to do about it.*

---

## Part A — Theory & reference

### What it does

The pipeline is **describe → attribute → prescribe**:

1. **describe** — count the recompilations in the session.
2. **attribute** — parse each failed guard's text into a *category* + a *dynamic axis*, and group
   recompiles by `(category, axis)` so "dim 0 keeps going dynamic" is one cluster, distinct from
   "dim 1 keeps going dynamic".
3. **prescribe** — turn each actionable cluster into one rejectable, evidence-linked suggestion.

The analyzer is a pure, CPU-only pass over the artifact — no PyTorch, no model execution. You
collect once (where torch + the workload live) and analyze anywhere (laptop, CI runner).

### Input: the three collection modes

`recompile-summary` consumes a `.cls.json`. You produce one with the collector, which has three
modes depending on what you already have:

| Mode | Source | API | When to use |
|---|---|---|---|
| **A — logs** | a `TORCH_LOGS=recompiles` text capture | `RecompileCollector.from_logs(path)` | You already ran with `TORCH_LOGS=recompiles` and have the log. Lowest friction. |
| **B — tlparse** | a [`tlparse`](https://github.com/ezyang/tlparse) output directory | `RecompileCollector.from_tlparse(dir)` | You run `tlparse`; gives richer structured graph data. (Recompile guard text may be absent in some tlparse outputs — see Limitations.) |
| **C — dynamo.explain** | programmatic `torch._dynamo.explain(...)` | `RecompileCollector.from_dynamo_explain(...)` | A quick in-script check while torch is already in-process. |

> A one-command `cl collect` CLI wrapper and a `cl.session()` hero flow are planned (Phase 7).
> Today the collector is the Python API shown in **Part B**; the analyzer (`cl recompile-summary`)
> is fully wired.

### Guard categories

Each recompile's failed guard is parsed into one category:

| Category | Means | Suggestion |
|---|---|---|
| `size` | a tensor dimension changed size (`tensor 'x' size mismatch at index 0`) | `mark_dynamic(x, 0)` / `dynamic=True` |
| `dtype` | a tensor's dtype changed across calls | pin the dtype (cast once) upstream of the compiled region |
| `stride` | a tensor was non-contiguous on some calls (layout changed) | insert `x = x.contiguous()` before the compiled region |
| `other` | an unrecognized guard (a value/bool guard like `n == 1`, or a symbolic shape guard) | none — integer literals are canonicalized to `<int>` so structurally-identical guards still group, but no confident fix is emitted |

`other` deliberately yields **no** suggestion rather than a vague one — a noisy tool gets disabled.

### Findings shape

The analyzer returns a `RecompileFindings` (this is exactly what `--format json` serializes):

- `total_recompilations` — every recompile event observed.
- `guard_categories[]` — one per `(category, axis)` cluster, ordered by descending count:
  - `category`, `count`, `canonical_guard` (the shared template), `axis` (e.g. `x[0]`),
    `observed_values[]` (distinct `previous → new` transitions, e.g. `4 → 8`).
- `top_suggestions[]` — `{ text, evidence }`, where `evidence` traces back to the cluster (the
  suggestion is never invented; it points at its data).

For the on-disk artifact's `recompilations[]` shape (`recompilation_id`, `failed_guard`,
`previous_value`/`new_value`, …), see the binding schema [`schema/v0.5.0.json`](../../schema/v0.5.0.json).

---

## Part B — Examples

The repository ships example artifacts under [`schema/examples/`](../../schema/examples/), so the
analyzer commands below run as-is from the repo root.

### 1. Single-session summary (Markdown — the default)

```bash
cl recompile-summary session.cls.json
```

For a session where dim 0 of `x` kept changing size:

```
## Recompile Summary

Total recompilations: 1
Triggered by: 1 distinct guard category
  └ size · `x[0]` (1 event — values 4→8)

### Top suggestions

1. `x` dim 0 keeps changing size — mark it dynamic to stop the recompiles: `torch._dynamo.mark_dynamic(x, 0)` (or compile with `dynamic=True`).
   _evidence: size cluster on x[0] — 1 recompile(s) (values 4→8)_
```

A mixed session (size + dtype + stride) clusters each axis separately and ranks the suggestions:

```
## Recompile Summary

Total recompilations: 3
Triggered by: 3 distinct guard categories
  ├ dtype · `x` (1 event — values Float→Double)
  ├ size · `x[0]` (1 event — values 8→16)
  └ stride · `x[0]` (1 event — values 4→1)

### Top suggestions

1. `x`'s dtype changes across calls, forcing a fresh compile each time — pin it (cast once) upstream of the compiled region.
2. `x` dim 0 keeps changing size — mark it dynamic to stop the recompiles: `torch._dynamo.mark_dynamic(x, 0)` (or compile with `dynamic=True`).
3. `x` is non-contiguous on some calls — insert `x = x.contiguous()` before the compiled region so its layout is stable.
```

### 2. Machine-readable JSON

```bash
cl recompile-summary session.cls.json --format json
```

Emits the `RecompileFindings` struct verbatim — pipe it into `jq` or a CI script:

```bash
cl recompile-summary session.cls.json --format json | jq '.total_recompilations'
```

`--format text` is also available for terminals that can't render Markdown.

### 3. Regression diff (CI: did this change make recompiles worse?)

Compare the current session against an earlier baseline:

```bash
cl recompile-summary head.cls.json --baseline main.cls.json
```

```
## Recompile Diff (vs baseline)

Added: 1 · Grown: 0 · Removed: 0

### Added — new recompiles this change introduced
  └ size · `x[0]` (1 event — values 32→24)
...
```

`added` = recompile axes this change *introduced*, `grown` = same axis with a higher count,
`removed` = fixed. Suggestions in diff mode are **regression-anchored** ("this change introduced N
recompiles on `x[0]` → mark_dynamic") — which the before/after diff makes defensible.

### 4. Producing the artifact (Python collector API)

```python
from compile_lens.collectors.recompile import RecompileCollector

# Mode A — a TORCH_LOGS=recompiles capture you already have:
#   TORCH_LOGS=recompiles python train.py 2>&1 | tee recompiles.log
collector = RecompileCollector("session.cls.json")
collector.from_logs("recompiles.log")
collector.finalize()        # writes session.cls.json

# Mode B — a tlparse output directory:
# collector.from_tlparse("tlparse_out/")

# then analyze it:
#   cl recompile-summary session.cls.json
```

### 5. GitHub Action integration (template)

Gate a PR on *not introducing* new recompiles, by diffing head against `main`:

```yaml
# .github/workflows/recompile-guard.yml
- name: Recompile regression check
  run: |
    cl recompile-summary head.cls.json --baseline main.cls.json --format json > diff.json
    added=$(jq '.added | length' diff.json)
    if [ "$added" -gt 0 ]; then
      echo "::error::This PR introduced $added new recompile cluster(s)"
      cl recompile-summary head.cls.json --baseline main.cls.json   # human-readable in the log
      exit 1
    fi
```

---

## Part C — FAQ, limitations & failure cases

### When should I use Mode A vs B vs C?

- **Mode A (logs)** — default. You already have (or can cheaply produce) a `TORCH_LOGS=recompiles`
  capture. No extra tooling.
- **Mode B (tlparse)** — you want the richer structured graph data tlparse provides, or you already
  run tlparse in your pipeline.
- **Mode C (dynamo.explain)** — a quick programmatic check inside a script where torch is already
  loaded.

### What happens if PyTorch nightly changes the recompile log format?

Mode A's parser is hand-written against the **current** `TORCH_LOGS=recompiles` text shape (no
regex — see the analyzer source). If a future torch changes the wording:

- guards it no longer recognizes fall into the **`other`** category (still counted and grouped, but
  no confident suggestion) — the tool degrades gracefully rather than crashing or lying;
- the fix is to update the Mode A parser, or switch to **Mode B (tlparse)**, whose structured input
  is less sensitive to log-text wording.

This is also why the test suite pins captured fixtures: a parser change shows up as a fixture diff.

### Known limitations

- **Malformed / value-less guards.** A recompile whose guard has no parseable expression is still
  counted in `total_recompilations` but contributes no cluster (skipped, not crashed). A guard that
  carries no before/after value (e.g. a bool guard `G['flag'] == True`) clusters by axis with an
  empty `observed_values`.
- **Mode B guard text.** Some `tlparse` outputs record a recompile without the failed-guard
  expression; those recompiles are counted but can't be attributed to a category/axis. Prefer Mode A
  when you need full attribution.
- **Suggestions are hints, not patches.** Every suggestion is rejectable and carries an `evidence`
  trace back to its cluster; the tool never edits your code or auto-applies a fix.
- **Source attribution** (which `.py` line owns the recompiling tensor) is not surfaced in the
  findings yet.

---

## Related

- [`schema/v0.5.0.json`](../../schema/v0.5.0.json) — the binding artifact schema.
- [`CLS-E0002`](../08_errors/cls-e0002.md) — what a malformed `.cls.json` reports.
- ADR-029 (`docs/02_design_decisions/adr-029-recompile-schema-deferral.md`) — why source
  attribution is derived by the analyzer rather than stored in the schema.
- Source: [`crates/cls-analyzer/src/recompile.rs`](../../crates/cls-analyzer/src/recompile.rs).
