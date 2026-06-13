# ADR-026: `cl.session()` is a factory + context manager with selective probes

- **Status**: Accepted
- **Date**: 2026-06-11
- **Deciders**: project maintainer
- **Related**: `design.md` §9.2 (the hero form); P7 (the Python session is the source of truth, the CLI is a thin wrapper over it); ADR-022 (typed errors — an unknown probe name is a typed argument error, not a silent no-op).

## Context

The hero form is the first thing a new user touches: roughly six lines of Python that run the
collectors over a `torch.compile` and produce one self-contained HTML report. `cl.session()` is
that entry point, and it is also the **source of truth** (P7) — `cl session report` is a thin
wrapper over the same API, so the shape chosen here is frozen into the v0.5.0 surface and into the
CLI's argument shape.

Two facts frame the choice:

1. **The default cost is asymmetric across probes.** A recompile probe is nearly free; an
   iterations probe is cheap; but a *divergence* probe needs both an eager and a compiled copy of
   the model and runs the work twice, and a *diff* needs a second session to compare against. A
   default that turns everything on would make the very first six-line example slow, which is
   exactly the wrong first impression.
2. **The CLI must map onto this API cleanly.** Whatever shape `cl.session()` takes, `cl session`
   has to express it as flags without drifting, or the two surfaces diverge over time.

A *pseudo-criterion* to name and set aside: the "fluent" reading of a chained builder
(`cl.session().recompile().diff()`) looks attractive for discoverability, but it is the shape that
maps *worst* onto CLI flags, so its readability advantage is paid back immediately at the wrapper.

## Decision

`cl.session()` is a **factory function returning a context manager**, with **selective probes**:

```python
cl.session(
    probes: set[str] = {"recompile", "iterations"},   # opt into "diff" / "kernels" / "divergence"
    output_dir: Path = ".compile-lens/",
    redaction_policy: RedactionPolicy = DEFAULT_STRICT,
) -> Session
```

- The **default probe set is `{"recompile", "iterations"}`** — the two cheap probes. The expensive
  ones (`divergence`, which needs two model copies; `diff`, which needs a second session) are
  opt-in, so the default six-line example stays fast.
- Probes are a **`set[str]`**, not an enum, so the CLI can pass `--probes recompile,iterations`
  straight through. An unknown probe name is a typed argument error, not a silent no-op.
- It is a lowercase **factory** (`cl.session()`), not the `Session` class exposed directly, so the
  hero example reads as lightly as possible; internally it constructs a `Session`.
- `__enter__` mounts the collectors; `__exit__` finalizes and writes the `.cls.json` (applying the
  redaction policy); `Session.report(output=None)` aggregates the result into the HTML report.

The v0.5.0 surface above is **frozen**. New probe names may be added later without breaking the
API (a forward-compatible extension); promoting `divergence`/`diff` into the default is a future
decision, not part of v0.5.0.

## Consequences

- The default hero example is fast and clean (`with cl.session() as s: model(x)`), and a user opts
  into cost explicitly.
- `cl session report` is a thin wrapper: its `--probes` flag is the same set, so the two surfaces
  cannot drift on probe selection.
- Adding a probe is additive; no existing call breaks.
- A user who wants everything must spell it out — a deliberate trade of brevity-at-max for a fast
  default.

## Alternatives considered

Scored on a weighted matrix (weights sum to 10; every option scored on the same 0–10 ruler).

| Dimension (weight) | (a) no-arg, all on | **(b) selective probes** | (c) fluent builder | (d) explicit class |
|---|---|---|---|---|
| First-use clarity / six-line cleanliness (3.0) | 10 | 9 | 7 | 6 |
| Default cost is controllable (2.5) | 3 | 9 | 8 | 8 |
| Selective mounting (2.0) | 2 | 9 | 9 | 9 |
| Maps cleanly onto the CLI wrapper (1.5) | 8 | 9 | 4 | 8 |
| Pythonic / discoverable (1.0) | 8 | 8 | 6 | 7 |
| **Weighted total (/10)** | **6.15** | **8.90** | **7.10** | **7.50** |

- **(a) no-arg, all probes on** — the cleanest possible six lines, but uncontrollable default cost
  (divergence runs the model twice) and no selectivity. Loses on the two highest-weight dimensions
  after clarity.
- **(c) fluent builder** (`cl.session().recompile().diff()`) — reads well in Python, but a chained
  builder is the hardest shape to express as CLI flags, so it scores lowest on wrapper-mapping and
  drags the total down.
- **(d) explicit `cl.Session(...)` class** — fully explicit and maps fine to flags, but exposing the
  capitalized class is heavier than a factory for the hero's six-line first impression.

(b) wins clearly (8.90 vs 7.50 / 7.10 / 6.15). The only assumption that could revive (a) is
weighting first-use clarity far above everything and zeroing default-cost; even then (b) trails on
clarity by a single point — `cl.session()` is still callable with no arguments — so (b) holds.
