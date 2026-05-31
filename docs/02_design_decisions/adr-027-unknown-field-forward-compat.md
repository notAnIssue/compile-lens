# ADR-027: Forward-compatible unknown-field capture (cross-language)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Deciders**: project maintainer
- **Related**: ADR-021 (schema layout). Implemented in the `cls-schema` crate (serde) and `python/compile_lens/_schema.py` (pydantic); guarded by `test_unknown_field_handled` in both.

> Numbered 027 (not 022) by deliberate choice: 022–026 are reserved for planned section
> ADRs (error handling, recompile migration, Tool 2, hero form). This ADR emerged out of
> plan during the initial schema bindings work, so it takes the next slot past the reserved block.

## Context

`.cls.json` is the **sole cross-language contract** between the Python collectors and the
Rust analyzers — no PyO3, no IPC, only a file on disk (design.md §5.4). The schema sets
`additionalProperties: true` at **every** level, and D6 (collector generosity) plus
design.md §2.3 establish that the field set only grows — especially session metadata
(`gpu_arch`, `driver_version`, `nccl_version`, …) and the top-level structure (new record
arrays). A deployed binding will therefore routinely meet keys its struct/model does not
declare.

Two questions follow, and they must be answered the **same way in both languages**, because
the cross-language round-trip test requires Rust and Python to agree byte-for-byte:

1. When a binding meets an unknown key, does it **preserve** it through a read → write
   cycle, or **drop** it?
2. Is that behavior **uniform** across the schema, or **selective** by location?

**Pseudo-criterion, rejected:** "preserve everything, to be safe." Preservation is not
free — every struct that preserves needs a catch-all field (a serde `flatten` map / a
pydantic `extra="allow"` model), which doubles the type surface and retains data no
consumer reads. Safety here is about *not losing growth at the points that grow*, not about
hoarding every byte everywhere.

## Decision

**Capture unknown keys only at the two growth points — the top-level artifact
(`ClsArtifact`) and the session object (`Session`) — and drop them everywhere else. The
line is drawn on the same two types in both languages.**

- **Rust**: `ClsArtifact` and `Session` carry `#[serde(flatten)] extra: IndexMap<String, Value>`;
  every other struct uses serde's default (an unknown key is consumed via `IgnoredAny` and
  never stored, so it is absent on re-serialize).
- **Python**: `ClsArtifact` and `Session` set `model_config = ConfigDict(extra="allow")`
  (unknowns kept in `__pydantic_extra__` and re-emitted); every other model uses
  `extra="ignore"` (the default).
- The distinction is **per type, not per JSON depth** — e.g. an unknown key directly under
  `session` is captured, but an unknown key inside `session.compile_config` is dropped,
  because `CompileConfig` is its own type without capture.
- Each language ships `test_unknown_field_handled`, asserting: an unknown key at
  top-level / `session` survives a round trip; an unknown key inside a deep record (e.g. a
  kernel) is accepted on read but absent on re-emit. **These paired tests — not convention —
  are what keep the line identical across the two languages.**

(Serialization mechanics that make a preserved key actually re-emit — serde
`skip_serializing_if` vs. pydantic `model_dump(exclude_unset=True)` — are governed by the
schema's optional/nullable rule, documented in the discussion notes; this ADR governs only
*where* capture happens.)

## Consequences

- **Positive**:
  - Forward compatibility lands exactly where the schema grows (D6): a collector running
    ahead of an analyzer does not lose new session / top-level fields on a round trip.
  - Cost is two catch-all fields, not ~24 — deep record types stay clean.
  - The two highest-value properties (growth-point fidelity + cross-language agreement) are
    both satisfied, and verified by the paired tests.
- **Negative / costs**:
  - Unknown keys inside deep records are silently dropped on re-emit. Accepted: analyzers
    compute *new* output rather than round-tripping deep input, so deep preservation has no
    current consumer.
  - The line must be kept synchronized across two languages — a real maintenance coupling,
    discharged by the paired `test_unknown_field_handled` tests rather than trusting prose.
- **Follow-ups**:
  - If a deep record later needs preservation, add the capture field to that one
    struct/model **in both languages** and extend both tests — a backward-compatible change.
  - Any new top-level- or session-level type inherits the growth-point policy by default.

## Alternatives considered

- **Option A — capture everywhere**: a catch-all field on every struct/model. Maximum
  fidelity, but ~24 catch-all fields per language and retention of deep data nothing reads.
- **Option B — capture nowhere**: serde default / `extra="ignore"` uniformly; all unknowns
  dropped. Simplest, but loses growth precisely where D6 says it happens (session / top-level).
- **Option C — capture at the two growth points only** (chosen).

### Weighted decision matrix

Criteria weighted to sum to 10; every option scored 0–10 on the same scale; weighted
contribution = `weight × (raw/10)`.

| Criterion (weight) — why this weight | A everywhere | B nowhere | C growth-points ✅ |
|---|---|---|---|
| **Forward-compat fidelity at growth points (3.0)** — the reason to capture at all; D6 says session/top-level grow | 10 (3.00) | 2 (0.60) | **9 (2.70)** |
| **Cross-language consistency / round-trip safety (2.5)** — Rust and Python must drop/keep on the same line or the round-trip suite diverges | 8 (2.00) | 8 (2.00) | **9 (2.25)** |
| **Simplicity / low maintenance (1.5)** — catch-all field count, cognitive load | 2 (0.30) | 10 (1.50) | **8 (1.20)** |
| **YAGNI — no speculative deep retention (1.5)** — analyzers don't round-trip deep input | 2 (0.30) | 9 (1.35) | **9 (1.35)** |
| **Implementable cleanly in serde + pydantic (1.5)** | 7 (1.05) | 10 (1.50) | **9 (1.35)** |
| **Weighted total / 10** | **6.65** | **6.95** | **8.85** |

**Readout**: C wins (8.85 vs B 6.95 vs A 6.65), dominating the two highest-weighted criteria
(growth-point fidelity + cross-language consistency = 5.5 / 10 of the weight). A buys maximal
fidelity at ruinous boilerplate and speculative retention; B is simplest but fails the core
forward-compat need. The result would flip to B only if session / top-level were *not*
expected to grow — but D6 and design.md §2.3 state explicitly that they do.
