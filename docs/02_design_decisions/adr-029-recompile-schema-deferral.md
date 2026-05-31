# ADR-029: Defer Tool 1 schema-field refinement to v0.6

- **Status**: Accepted
- **Date**: 2026-05-31
- **Deciders**: project maintainer
- **Related**: ADR-021 (schema layout); ADR-027 (unknown-field capture at growth points only); [`schema/v0.5.0.json`](../../schema/v0.5.0.json); `design.md` §1.1 (release mapping) and §3 (Tool 1).

## Context

Tool 1 (the recompile aggregator) is Phase 1's deliverable. Before any
implementation lands, the schema fields it produces and consumes must be
audited against its responsibilities:

- Collection in three modes — `TORCH_LOGS=+recompiles` raw text, `tlparse`
  structured output, and `torch._dynamo.explain` programmatic output.
- Analysis — guard expression clustering, top-suggestion ranking, source
  attribution.
- Rendering — markdown / json / text outputs of the same finding set.

The relevant schema types in v0.5.0 (shipped at `v0.5.0-alpha.0`) are
`Recompilation`, `FailedGuard`, `CompiledGraph`, `Iteration`, and the nested
`GuardEvaluation`. The question is whether they carry enough for the work
above, or whether Phase 1 needs a `v0.5.1` minor bump first.

A consequential constraint: ADR-027 captures unknown keys only on
`ClsArtifact` and `Session`. Deeper records (including `FailedGuard`) drop
unknown keys on re-serialization. There is no "park new fields in `extra`
until v0.6" path for the deep types; whatever the schema commits to is what
gets round-tripped.

**Pseudo-criterion, rejected**: *"schema cleanliness now beats schema cleanliness
later."* A schema-minor bump is a public-API surface promise under SemVer.
v0.5 is pre-MVP — Tool 2a and the Hero report still have not shipped. Buying
SemVer lock-in before the MVP fully runs is the YAGNI-trap pattern Phase 0
explicitly closed.

### Audit results

| Tool 1 need | Currently in schema | Sufficient? |
|---|---|---|
| Identify each recompile event | `Recompilation.recompilation_id` + `compiled_function_id` + `trigger_reason` | yes |
| Failed-guard details | `FailedGuard.{guard_id, expression, previous_value, new_value}` | yes |
| When the recompile happened | `Recompilation.occurred_at_step` + `wall_clock_ms` | yes |
| Cross-reference to compiled graph | `Recompilation.compiled_function_id` + `CompiledGraph.compiled_function_id` | yes |
| Per-step guard evaluations (for Mode B replay) | `Iteration.guard_evaluations[]` | yes |
| **Source attribution for a failed guard** | *not present* — `SourceLocation` exists as a type and is used on `LintFinding`/`GraphBreak`, but not on `FailedGuard` | **no** |
| Guard category hint (shape / value / type) | *not present* | no — but optional, derivable from the expression |

One field is genuinely missing for collector-emitted attribution:
`FailedGuard.source_location`. One more would be additive but derivable:
`FailedGuard.category`. Everything else Tool 1 needs is already present.

## Decision

**Tool 1 ships against the v0.5.0 schema unchanged. Field additions for the
recompile path are deferred to the v0.6 schema bump.**

Specifically:

- Source attribution for a failed guard is **derived in the analyzer**, by
  parsing `FailedGuard.expression` and the surrounding `Recompilation` /
  `CompiledGraph` context. It is documented as a best-effort feature in the
  Tool 1 docs page.
- No `cls-schema-migrate` step is added for a v0.5.0 → v0.5.1 transition.
  The detect-and-refuse skeleton stays as it is.
- A wishlist of fields, recorded below, travels into the v0.6 schema work
  with empirical justification from real Tool 1 sessions.

### Wishlist for v0.6 (captured, not committed)

When the v0.6 schema work opens (and its own ADR is written), these are the
candidates from this audit. They are intentionally **not** in this ADR's
*Decision* section — they are recorded so the v0.6 author has a starting
list, not so they ship without re-evaluation:

- `FailedGuard.source_location: SourceLocation`. Lets the collector own
  attribution, so the analyzer is not coupled to PyTorch's log format for the
  attribution path (only for Mode A parsing, which is the irreducible
  coupling). Reuses the existing `SourceLocation` type — same shape as
  `GraphBreak.location`.
- `FailedGuard.category: string` (open string, not enum, per ADR-021 D6
  forward-compat rule). Vocabulary suggestion: `shape` / `value` / `type` /
  `callable` / `unknown`. Pure clusterer hint; not load-bearing.
- `Recompilation.frame_id: string` (optional). Currently approximated by
  `compiled_function_id`. Worth revisiting only if Tool 1 sessions surface a
  case where the approximation breaks.

## Consequences

- **Positive**.
  - No migration churn during Phase 1. The `v0.5.0` → `v0.6` schema jump is
    one SemVer step rather than two, and the `cls-schema-migrate` skeleton
    keeps its current detect-and-refuse shape until there is one real
    migration to write.
  - Tool 1 implementation discovers what is actually missing under load,
    rather than fixing fields we *guess* are missing. v0.6 ships with
    evidence.
- **Negative / costs**.
  - Source attribution in Tool 1 is best-effort. The Tool 1 docs page must
    say so, and the limitations section must include a worked example of a
    case where attribution fails or is ambiguous.
  - The wishlist lives in this ADR. If the v0.6 schema author forgets to
    consult ADR-029, the audit's findings are lost. The v0.6 schema ADR's
    *Related* section will need to point back here.
- **Follow-ups**.
  - The remaining Phase 1 work — clustering, top-suggestion generation,
    rendering — consumes `FailedGuard.expression` as-is and produces
    `SourceLocation` from analyzer logic. No schema dependency.
  - `docs/03_tools/recompile_summary.md` (Phase 1 deliverable) explicitly
    documents the analyzer-derived attribution and its failure modes.

## Alternatives considered

- **Option A — Status quo, no recorded wishlist.** Cheapest now; loses the
  audit's findings. If the wishlist is not written somewhere, v0.6 likely
  rediscovers it from scratch.
- **Option B — `v0.5.1` minor bump now**, adding `FailedGuard.source_location`
  and `FailedGuard.category`. Cleanest schema state. But it commits SemVer
  surface to fields that have not been tested under Tool 1 load yet — the
  exact YAGNI failure mode Phase 0 closed. It also forces a real `v0.5.0` →
  `v0.5.1` migration in `cls-schema-migrate` before there is any consumer to
  validate it against.
- **Option C — Defer to v0.6, record the wishlist in this ADR** (chosen).

### Weighted decision matrix

Criteria weighted to sum to 10; every option scored 0–10 on the same scale.

| Criterion (weight) — why this weight | A status-quo | B v0.5.1 bump | C defer + record ✅ |
|---|---|---|---|
| **No-premature-commitment / YAGNI (3.0)** — schema is API surface and v0.5 is pre-MVP; the Phase 0 closeout explicitly retired this trap | 8 (2.40) | 3 (0.90) | **9 (2.70)** |
| **Discoverability for the v0.6 schema work (2.5)** — without a recorded wishlist the audit's findings are wasted | 3 (0.75) | 9 (2.25) | **9 (2.25)** |
| **Implementation overhead during Phase 1 (2.0)** — migration step, binding update, fixture re-generation | 10 (2.00) | 4 (0.80) | **10 (2.00)** |
| **Tool 1 capability ceiling (1.5)** — collector-emitted attribution vs analyzer-derived | 6 (0.90) | 9 (1.35) | **6 (0.90)** |
| **Empirical evidence preserved for v0.6 (1.0)** — Tool 1 actually run before the schema gets minor-bumped | 6 (0.60) | 3 (0.30) | **10 (1.00)** |
| **Weighted total / 10** | **6.65** | **5.60** | **8.85** |

**Readout**: C wins by dominating the two highest-weighted axes —
YAGNI *and* discoverability for v0.6. It pays a real Tool 1 capability cost
(analyzer-derived attribution is best-effort and the docs page must say so),
but the cost is acceptable because the analyzer was always going to need
PyTorch-version coupling for Mode A parsing anyway; the same coupling is
extended one step to cover attribution. Option B is the cleanest schema
state but buys SemVer commitment without evidence — exactly what Phase 0's
`v0.5.0` closeout retired. Option A is the lowest-effort path on day one and
the highest-cost path at v0.6, because the wishlist evaporates.

The result would flip to B only if *Tool 1 capability ceiling* were
re-weighted to ≥ 3.5 — which would imply collector-emitted attribution is
critical-path. It is not: source attribution is a hint, not a guarantee,
and the docs page commits to that framing.
