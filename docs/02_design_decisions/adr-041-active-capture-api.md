# ADR-041: `cl.capture()` — an active, tool-driven capture alongside the passive `cl.session()`

- **Status**: Accepted
- **Date**: 2026-06-21
- **Deciders**: project maintainer
- **Related**: ADR-026 (the `cl.session()` probe API); ADR-006 (Python↔Rust boundary is a JSON file); ADR-032 (each tool emits substrate the report aggregates); ADR-040 (diff into the hero report).

## Context

The hero report has seven sections, and they are fed by *different* collectors: graph capture
(Tool 2a diff / Tool 6 fusion), the per-iteration cache probe (Tool 2b), the recompile log
(Tool 1), the source lint scan (Tool 4), the eager-vs-compiled divergence localizer (Tool 3), and
the kernel roofline (Tool 5). The README long described this as "one capture → six analyzers."

That framing was ahead of the code. `cl.session()` (ADR-026) is a **passive context manager**:

```python
with cl.session() as s:
    output = model(input)
```

It only sees what crosses stderr during the block, so the single probe it implements is `recompile`
(it tees torch's recompile log). It never receives `model` or the inputs, so it *structurally cannot*
trace the graph, re-run for the cache probe, vary shapes for a recompile storm, or run eager beside
compiled for divergence. ADR-026 even records this: `iterations` / `diff` / `divergence` / `kernels`
are recognized-but-unbuilt probes. So a report built from a real `cl.session()` capture lights up at
most one section — and the committed hero was actually produced by reaching past `cl.session()` to
the lower-level `CompileArtifactCollector` directly, which is why it only ever showed two sections.

The honest options were: keep underselling (one real capture → ~2 sections), fabricate the rest
(rejected outright — that is the failure mode this project exists to avoid), or make one call
genuinely drive every collector.

## Decision

Add **`cl.capture(model, example_input, *, base=…, vary_inputs=…, source=…, check_divergence=…)`** —
an *active* orchestrator. You hand it the model and a representative input, and one call runs each
built collector and assembles a single `.cls.json` the report renders end to end:

| Section | How `cl.capture()` produces it |
|---|---|
| Fusion (6) / IR-diff (2a) | graph capture via `CompileArtifactCollector` (aten FX graph); `base=` captures a baseline graph to diff against |
| Cache stability (2b) | `iterations` repeated same-shape runs — shows the cache was stable, or a real stale-cache bug if one exists |
| Recompile (1) | the model run across `vary_inputs`, with torch's recompile log teed and parsed |
| Lint (4) | a static AST scan of `source` |
| Divergence (3) | eager vs a deep-copied compiled model, first divergent layer localized — **empty when they agree** |

`cl.session()` is **kept unchanged** as the lightweight, passive, recompile-only form. Roofline
(Tool 5) is deliberately excluded: it needs a *measured* GPU kernel profile, which a CPU capture
cannot honestly produce, so it is collected separately against real hardware.

Two implementation points carry weight:

- **Recompiles are captured with `backend="eager"`.** Recompilation is a dynamo concept (a guard
  failure forces a re-trace regardless of backend), so the recompile probe needs no inductor. This
  keeps it fast and, crucially, isolated from any inductor pass a caller installed to study a
  *different* tool (e.g. a miscompiling custom pass set up to exercise divergence must not perturb
  the recompile count).
- **Divergence never fabricates.** A clean model returns no divergence record; only a genuine
  eager≠compiled gap produces one.

## Alternatives considered

- **Extend `cl.session()` to take the model (rejected).** It would either break the elegant passive
  form (the whole point of the context manager is that you run *your own* model in the block) or
  require intercepting that in-block call, which torch does not support cleanly. It also churns the
  ADR-026 frozen contract. A separate active entry leaves the passive one intact and honest about
  what each does.
- **Leave the README's "one capture" framing and stitch artifacts only in the demo (rejected).**
  Presenting separately-captured arrays as if one call produced them is the same overclaim as naming
  a kernel that does not exist. If the docs say "one call," one call must do it.
- **A CLI-first `cl capture` subcommand (deferred).** The Python API is what the hero and the
  six-line pitch use; a CLI wrapper over the same orchestration can follow without re-deciding this.

## Consequences

- The hero is a genuine one-call capture: `cl.capture(...).report().save_html(...)` lights up five
  sections with real findings (recompile, IR-diff, divergence, lint, fusion), an honest "cache
  stable" check, and an honest "roofline needs a GPU capture" note.
- The README's capture framing is corrected to describe `cl.capture()` (the active form) and
  `cl.session()` (the passive recompile-only form) for what each actually does.
- `cl.capture()` composes the existing collectors and adds no new analysis; each section's honesty
  rules (ADR-032 substrate, ADR-034 divergence, ADR-036 lint) are inherited unchanged.
