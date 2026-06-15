# ADR-024: Inline node-level FX-graph representation for diffing

- **Status**: Accepted
- **Date**: 2026-06-09
- **Deciders**: project maintainer
- **Related**: ADR-021 (schema layout); ADR-027 (unknown-field capture at growth points only); ADR-029 (defer Tool 1 schema-field refinement to v0.6); ADR-032 (the `.cls.json` as a self-contained, queryable substrate); [`schema/v0.5.0.json`](../../schema/v0.5.0.json); Tool 2a (compile-diff).

## Context

Tool 2a diffs two compiled FX graphs (a base and a head compile) to report which
nodes a change **added, removed, or modified**. An FX graph is the computation
graph `torch.compile` traces from model code: a sequence of operation nodes with
dependency edges. To diff it, the analyzer needs node-level structure — each
node's `op_type` (e.g. `aten.matmul`), its **ordered** inputs (which upstream
nodes feed it, in order — order matters because `sub(a, b)` ≠ `sub(b, a)`), and
its attributes (a constant's value, a dim parameter).

The shipped schema does not carry this. `compiled_graphs[]` has only
`fx_graph_path`: a path string with no node-level structure (Tool 1 never needed
the graph shape, so it was never built). The decision is **what form the
node-level structure takes in the `.cls.json`** so the Rust analyzer can read it.

A constraint frames the whole choice: compile-lens's identity is that a single
`.cls.json` *is* a self-contained, queryable substrate for one compile (ADR-032),
and the acceptance target is PyTorch's official **small** models.

A *pseudo-criterion* to name and discard up front: "diff speed / query
performance." It looks relevant but is not — diff cost is set by the WL-signature
algorithm, and all storage options deserialize into the same in-memory graph
before that algorithm runs, so storage form barely moves it. Scoring it would
only dilute the axes that actually separate the options (self-containment, size).

## Decision

Add an optional `nodes: FxNode[]` array to `CompiledGraph`, with the minimum
node shape the diff needs:

- `id` — graph-unique node id; edges are encoded by other nodes referencing it in
  their `inputs`.
- `op_type` — the operation.
- `inputs` — **ordered** ids of upstream nodes. The ordering is load-bearing and
  must be preserved as a sequence, not a set, or operand-order regressions become
  invisible.
- `attrs` — free-form node attributes used for `modified` detection.

The structure is **inlined** into the artifact (not a side file). The field is
**optional and forward-compatible**: a Tool 1 artifact written before it existed
simply omits it, and an empty `nodes` is skipped on serialization (matching
ADR-027's tolerance of absent/unknown fields). `fx_graph_path` is retained for a
full debug dump; the diff consumes only `nodes`. No schema-version bump and no
migration: pre-V1, adding an optional field does not break existing artifacts
(consistent with the `cls-schema-migrate` "re-collect" stance).

## Consequences

- **Positive**: one `.cls.json` remains everything you need to diff — no risk of
  losing a side file in CI or when sharing. The analyzer just deserializes; there
  is no file-loading seam to get wrong. Both language bindings and the JSON Schema
  carry the same shape, with cross-language round-trip parity enforced in CI.
- **Negative / costs**: the artifact grows with graph size (inlined node JSON).
  Acceptable for the small-model target (hundreds to a few thousand nodes ≈ a few
  hundred KB); revisit if the hero phase ever diffs very large production graphs.
- **Follow-ups**: the collector (a later Phase 2 PR) serializes
  `torch.fx.GraphModule` into `nodes[]`; the WL-signature PR builds its in-memory
  graph from `nodes[]`.

## Alternatives considered

- **A — Inline `nodes[]`** (chosen): node-level JSON inside `compiled_graphs[]`.
- **B — External file**: keep `fx_graph_path` pointing at a separately serialized
  graph and define that file's format; the analyzer loads it.
- **C — Hybrid**: inline the minimal node structure the diff needs, keep the full
  dump behind `fx_graph_path` for debugging.

Weighted decision matrix (weights sum to 10; each option scored 0–10 on the same
scale, weighted contribution in parentheses):

| Criterion (weight) | A inline | B external | C hybrid |
|---|---|---|---|
| Self-contained / shareable / reproducible (3) | 9 (2.70) | 3 (0.90) | 8 (2.40) |
| Analyzer simplicity — no load seam (2) | 9 (1.80) | 4 (0.80) | 8 (1.60) |
| Artifact size / large-graph scaling (2) | 5 (1.00) | 9 (1.80) | 7 (1.40) |
| Collector serialization cost (1.5) | 7 (1.05) | 6 (0.90) | 6 (0.90) |
| Schema / migration cost (1.5) | 6 (0.90) | 8 (1.20) | 5 (0.75) |
| **Weighted total / 10** | **7.45** | **5.60** | **7.05** |

- **Weight justification**: self-containment is heaviest (3) because it is the
  product's identity (ADR-032) — getting it wrong is the most expensive error —
  but not 4+, since artifact size is a real constraint. Analyzer simplicity (2):
  a load seam is a long-term bug source, but lighter than identity. Size (2): A's
  genuine cost, given full weight but not above identity. Collector cost and
  schema/migration cost (1.5 each): one-time, similar across options, cheap to
  change.
- **Readout**: A wins (7.45) and dominates B on the heaviest axis.
- **B's fatal flaw**: the artifact is no longer self-contained — diffing needs
  both `.cls.json` files *plus* their side files, trivially half-lost in CI or
  sharing, which directly contradicts the substrate identity. Its size advantage
  cannot buy that back.
- **C's flaw**: not wrong, just not worth it — it pays for "keep two graph
  representations consistent" to save size that, at the small-model target, never
  hurts. Premature size optimization.
- **What would flip it**: if the target shifted from official small models to
  routinely diffing very large (tens of thousands of nodes) production graphs,
  A's inlined-size weakness would amplify and B/C's "keep the main artifact small"
  edge could overtake it. The acceptance target is small models, so A holds;
  revisit if the hero phase needs large-graph diffing.
