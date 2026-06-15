# ADR-039: Explicit string sentinels for non-finite f64

- **Status**: Accepted
- **Date**: 2026-06-15
- **Deciders**: project maintainer
- **Related**: ADR-021 (schema layout); the `cls-schema` crate; the `.cls.json` contract; Tool 3 (`max_abs_diff`) and Tool 5 (roofline numbers).

## Context

The `.cls.json` artifact is the sole cross-language contract: Python collectors write it, Rust
analyzers read it. Several `f64` fields hold *computed* values that can legitimately be non-finite —
most importantly `max_abs_diff` (Tool 3), which is **NaN exactly when the compiled model produces
NaN at a layer**: the strongest divergence signal the tool can report. Others (`estimated_speedup`,
the correlations, `arithmetic_intensity`) can reach ±∞ through a division by zero.

JSON has no NaN or Infinity literal. Both serializers in play silently map a non-finite `f64` to
`null`:

- `serde_json::to_string(&Some(f64::NAN))` → `"null"` (no error, no panic — verified);
- pydantic `model_dump_json()` of a NaN float → `null` (verified).

Because every such field is `Option<f64>` / `float | None`, that `null` is **indistinguishable from
"not computed"**. A NaN divergence and an absent measurement collapse to the same wire value, and
the signal is lost — silently, with no error to notice. (An earlier framing claimed serialization
*panics*; it does not. The real defect is the silent lossy conflation, which is worse, because
nothing surfaces it.)

## Decision

Encode a non-finite `f64` **explicitly**, as one of the JSON strings `"NaN"`, `"Infinity"`,
`"-Infinity"`. Finite values stay JSON numbers; `None` stays absent. The encoding round-trips
losslessly, and `null`/absent (= not computed) stays distinct from a sentinel (= computed, but
non-finite).

The policy applies **uniformly to every `Option<f64>` field** in the schema, via one reusable serde
adapter (`crate::non_finite::opt`, applied with `#[serde(with = …)]`). Uniformity is deliberate: a
per-field "this one can be non-finite, that one can't" rule is a judgment that ages badly — a field
assumed finite that later divides by zero silently re-introduces the bug. One rule, no exceptions,
nothing to get wrong. The two bare `f64` fields (`rtol`, `atol`) are user-set tolerances,
contractually finite, and are left unannotated.

## Alternatives considered

1. **Reject / fail-fast.** Make a non-finite value a hard error (a new `CLS-E…` code); the schema
   invariant becomes "all `f64` are finite". *Rejected:* a NaN `max_abs_diff` is a real finding, not
   a corruption — erroring the write throws away the very signal Tool 3 exists to surface, and pushes
   NaN-handling onto every producer.
2. **Keep `null`, document it.** Accept the status quo as intentional. *Rejected:* it keeps the
   lossy conflation the audit flagged; for a diagnostics tool, losing "diverged to NaN" is losing the
   headline result.
3. **Explicit string sentinel.** *Chosen:* lossless, never errors, and keeps non-finite distinct from
   absent — at the cost of a union type (`number | "NaN" | …`) on those fields, which consumers must
   accept.

## Consequences

- **Rust (this change).** A `serde(with)` adapter on all `Option<f64>` fields: finite → number,
  NaN/±∞ → sentinel, `None` → absent. Reading accepts number, sentinel, or null. Finite values
  serialize byte-for-byte as before, so existing round-trips and the determinism suite are
  unaffected.
- **Python read — already conformant.** pydantic v2 already coerces `"NaN"`/`"Infinity"` strings to
  `float` on input, so the Python reader needs no change.
- **Python write — follow-up.** pydantic currently emits `null` for non-finite. A field serializer
  must emit the sentinel instead, so that the *writer* of `.cls.json` (Python is the primary writer)
  actually produces the encoding. Until then, this change makes Rust *able* to read and write
  sentinels; the end-to-end value (a NaN surviving Python → Rust) lands with that follow-up.
- **JSON schema — follow-up.** `schema/v0.5.0.json` declares these fields `"type": "number"`; it
  should widen to also accept the sentinel strings, to keep the published contract in sync with the
  types.
