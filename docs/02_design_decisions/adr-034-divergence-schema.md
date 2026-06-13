# ADR-034: Divergence schema section (Tool 3 serialization contract)

- **Status**: Accepted
- **Date**: 2026-06-12
- **Deciders**: project maintainer
- **Related**: ADR-021 (schema layout: normalized top-level arrays); ADR-032 (describe → attribute; pass-level attribution); ADR-027 (unknown-field capture at growth points only); ADR-029 (schema additions during alpha); [`schema/v0.5.0.json`](../../schema/v0.5.0.json); `design.md` §8.3 (Tool 3).

## Context

Tool 3 produces two results at runtime — `DivergenceFindings` (first divergent layer, tolerances, layer count, `suggested_cause`) and `CausalAttribution` (which inductor pass, when disabled, removes the divergence). Until now these live only in memory; nothing persists them.

The next steps need them on disk: a view-only CLI (`cl divergence-view <session.cls.json>`) that renders a prior session without re-running torch, and the hero report that folds divergence into a unified view. Both require Tool 3's results to be serialized into `.cls.json`. But the artifact's schema (`schema/v0.5.0.json` and the Rust `cls-schema` bindings) has no divergence section — `ClsArtifact` carries `graph_breaks`, `recompilations`, `compiled_graphs`, `compile_phases`, `iterations`, `lint_findings`, `kernels`, `roofline_predictions`, but nothing for divergence, and `design.md` never specified its shape.

`.cls.json` is the *sole* cross-language contract (Python writes, Rust reads; the file on disk is the API). Getting a contract field wrong is expensive: both ends must change in lockstep, so this is settled schema-first — lock the contract in this change, then conform the Python producer and the Rust reader in later ones.

A pseudo-criterion to name and discard: "one run usually has only one divergence comparison" tempts a single-object shape. But basis cardinality (one vs many) does not decide array-ness here; *category* does — `lint_findings` also commonly holds one entry and is still an array, because it is an analysis result.

## Decision

Add a normalized top-level array `divergences: Vec<Divergence>` to `ClsArtifact`, a sibling of `lint_findings`:

- Each `Divergence` carries a `divergence_id` and mirrors the Python `DivergenceFindings` 1:1 (`first_divergent_layer`, `max_abs_diff`, `num_layers_compared`, `rtol`, `atol`, `suggested_cause`). Required: `divergence_id`, `num_layers_compared`, `rtol`, `atol`; the rest are absent-when-unknown.
- The causal attribution is **nested** as `Divergence.attribution: Option<DivergenceAttribution>`, not a separate id-joined array, because it is strictly subordinate to one finding (mirrors how `Session` nests `compile_config`).
- **No SemVer bump.** This is an additive, optional field (`skip_serializing_if` empty), backward-compatible at the data level and forward-compatible via `additionalProperties`. v0.5.0 is still alpha and being filled in section by section; no `cls-schema-migrate` function is required.

## Consequences

- **Positive**: divergence is consistent with every other analysis-result section (normalized array, ADR-021); the view CLI and hero report read it without torch; multiple divergence comparisons per artifact are representable without a future breaking change; the contract is locked before any producer/consumer is written.
- **Negative / costs**: the Python producer and the Rust reader must both conform to these field names; `divergence_id` is currently unreferenced (a record-id carried for convention-consistency and the hero report's likely linking, not a present join).
- **Follow-ups**: a later change has the Python divergence tool write `divergences[]`; a further one adds `cl divergence-view` rendering. `suggested_cause` equals `attribution.summary` when both are present — a documented, deliberate denormalization (the localizer may set `suggested_cause` without a full attribution).

## Alternatives considered

The core fork is **array vs single object**; a secondary one is **nesting vs id-joining the attribution** (decided: nest — a strict 1:0..1 subordinate relationship does not warrant a normalized join, consistent with `Session` nesting `compile_config`).

Weighted decision matrix for the array-vs-single fork:

1. **Schema-identity consistency** (weight 4): `ClsArtifact` is defined as "a session plus the normalized parallel record arrays"; every *analysis result* is an array. High weight because an inconsistent section is a contract smell every future reader pays for.
2. **Forward-compat for multiple comparisons** (weight 3): one artifact may record several eager-vs-compiled checks; growing a single object into an array later is a breaking change needing a migration.
3. **Simplicity / not over-building** (weight 2): the common case is one comparison per run.
4. **Semantic honesty** (weight 1, *pseudo-criterion*): "usually one" tempts the singular, but category, not cardinality, decides — discounted deliberately.

Scores are on one 0–10 scale; weighted contribution = `weight × (raw/10)`.

| Criterion (weight) | A: `Vec<Divergence>` | B: `Option<Divergence>` |
|---|---|---|
| Schema-identity consistency (4) | 9 (3.6) | 3 (1.2) |
| Forward-compat (3) | 9 (2.7) | 2 (0.6) |
| Simplicity (2) | 5 (1.0) | 9 (1.8) |
| Semantic honesty — pseudo (1) | 6 (0.6) | 7 (0.7) |
| **Total** | **7.9** | **4.3** |

**Readout**: A (array) wins decisively. B's fatal point is that it would be the only non-array analysis result in the schema, breaking the normalized-arrays identity and boxing in any future multi-comparison artifact behind a breaking change. The result would flip only if the schema's identity were "one flat result object per run" rather than normalized arrays — which it is not (ADR-021).
