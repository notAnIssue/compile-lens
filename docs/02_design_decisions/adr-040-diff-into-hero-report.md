# ADR-040: How a base→head compile diff enters the hero report

- **Status**: Accepted
- **Date**: 2026-06-17
- **Deciders**: project maintainer
- **Related**: ADR-006 (`.cls.json` as the Python↔Rust boundary); Tool 2a (`diff_graphs` in `cls-wl-diff`); the `cl session report` command; the `cls-report` crate.

## Context

`cl.session()` captures a **single** run and writes **one** `.cls.json`. But the regression question the report exists to answer — *did this change make `torch.compile` quietly worse?* — is intrinsically a comparison of **two** captures: a base and a head. The IR Diff (Tool 2a, `diff_graphs`) reports exactly what the head graph added, removed, or modified relative to the base, with match-coverage and anchor-uniqueness as quality gauges.

When `cl session report <session>` first shipped it rendered the single-session sections only; it was left unspecified how two captures feed one report. This ADR fixes that.

## Decision

Extend the existing command rather than add a new one:

```
cl session report <head> [--base <base>]
```

- `<head>` is the positional session; all single-session sections (metadata, recompile, cache stability, divergence, fusion, raw) render from it as before.
- `--base <base>`, when given, loads a second artifact, computes `diff_graphs(base, head)` → `IrGraphDiff`, and feeds it to the renderer.
- `cls_report::render` takes an `Option<&IrGraphDiff>`. `Some(diff)` renders a leading **IR Diff (base → head)** section, placed right after the metadata so the regression verdict is the first analytical thing a reviewer sees. `None` omits the section entirely — a single-session report has nothing to diff.
- The diff is computed CLI-side (the CLI already depends on `cls-wl-diff`); `cls-report` gains a `cls-wl-diff` dependency for the `IrGraphDiff` **type only**, and re-exports it so callers can name it.

## Alternatives considered

Scored on a weighted matrix (weights sum to 10; every option scored on the same 0–10 ruler):

| Dimension (weight — why it matters) | (a) `report <head> [--base]` | (b) `cl diff … --format json` piped into `report --diff-json` | (c) `session.diff_against(base)` Python method |
|---|---|---|---|
| One command produces the comparison (2.5 — a reviewer should judge the change in a single look; fewer commands is better) | 9 | 5 | 7 |
| Capture and compare stay separate responsibilities (2.5 — a session's job is to *capture*; comparing two historical artifacts is post-capture analysis) | 9 | 8 | 4 |
| Reuses the diff engine and CLI vocabulary (2.0 — `diff_graphs` already exists; `cl diff --base/--head` is already the vocabulary) | 9 | 7 | 6 |
| Minimal change to current signatures/architecture (2.0 — `cl session report <session>` already shipped) | 8 | 6 | 5 |
| CLI stays lean — flag count on `report` (1.0 — `report` will also grow `--gpu`/`--db`) | 6 | 6 | 8 |
| **Weighted total /10** | **8.5** | 6.45 | 5.75 |

**(a) wins decisively (8.5).**

- **(b)** is two commands plus an intermediate JSON file the user has to manage and plumb through. It keeps responsibilities clean and reuses the engine, but the extra steps and the temporary artifact drag down the one-look demo experience.
- **(c)** scores lowest on responsibility (4): the comparison happens *after* the `with` block — head is already captured, and we then diff it against a base *file* — so folding it into the capture session is semantically awkward, and the method would still have to shell out to the diff engine internally. It gives the session an API it shouldn't own.

**Sensitivity.** Halving the "responsibilities" weight and raising "CLI lean" toward 2.5 lifts (c), but its demo (7) and reuse (6) still trail (a); the ranking holds.

## Consequences

- **`render` signature.** Now `render(artifact, diff: Option<&IrGraphDiff>)`. Single-session callers (the CLI without `--base`, the tests) pass `None`. `IrGraphDiff` is re-exported from `cls-report`.
- **Dependency.** `cls-report` depends on `cls-wl-diff` for the type only; the diff itself is computed in the CLI, which already depended on `cls-wl-diff`. No new graph-matching logic enters the renderer.
- **Section placement.** The IR Diff leads (immediately after metadata) in `--base` mode and is absent otherwise — the regression view is the headline exactly when there is a regression to view.
- **Reversibility.** One CLI flag plus one optional render parameter. Switching later to (b) or (c) is a few lines; no architecture is locked by this choice. The diff inputs follow `cl diff`: the first compiled graph on each side, and an artifact with no compiled graph diffs as an empty graph rather than erroring.
