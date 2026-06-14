# ADR-037: Tool 6 (fusion detector) is a Rust analyzer over the serialized FX graph

- **Status**: Accepted
- **Date**: 2026-06-14
- **Deciders**: project maintainer
- **Related**: ADR-006 (Python↔Rust boundary is a JSON file, no in-process binding); ADR-024 (FX-graph node contract — `FxNode`); the `cls-wl-diff` / `cls-analyzer::lint` precedent (Rust analyzers over the serialized graph); the Tool 5 single-source decision.

## Context

Tool 6 (the CODA-style algebraic fusion-opportunity detector) finds `GEMM-Residual-RMSNorm-GEMM`
patterns in a `torch.compile` FX graph, judges foldability, estimates baseline-vs-fused HBM traffic,
and suggests an `a fused epilogue kernel` kernel — suggest-only, no graph mutation, no execution.

The matcher could be written in **Python** (operating directly on the live `torch.fx.Graph`) or in
**Rust** (operating on the FX graph already serialized into the `.cls.json`). The choice sets where
Tool 6 lives and how it relates to Tool 2.

Two facts decide it:

1. **Tool 2 already serializes the graph.** Its `CompileArtifactCollector` dumps the FX graph into
   `kernels`/`nodes[]` as `FxNode` records (ADR-024): each node has `op_type` (e.g. `aten.mm`), an
   ordered `inputs` list of upstream node ids (the edges — order is load-bearing), and an `attrs`
   bag. Pattern A needs exactly this: `op_type` to recognize `is_matmul` / `is_add` / `is_rms_norm`,
   and the edges to enforce the single-consumer topology (a node is single-consumer iff its id
   appears in exactly one other node's `inputs` — computable with one reverse index over `nodes[]`).
2. **The project's boundary is already drawn** (ADR-006): Python collects and writes the `.cls.json`;
   Rust analyzes it. `cls-wl-diff` (Tool 2a) and `cls-analyzer::lint` (Tool 4) already consume the
   serialized graph in Rust. A Python matcher would make Tool 6 the lone analyzer reaching back into
   live Python graph state — a second, divergent way to consume the FX graph.

## Decision

**Tool 6's matcher, cost model, and renderer are Rust, in a new `cls-analyzer::fusion` module**,
consuming the serialized `FxNode[]` from the `.cls.json` produced by Tool 2's collector. No new
collector and no re-trace: Tool 6 is another analyzer over the artifact, exactly like `lint` and
`roofline`. The cost model reuses the `cls-roofline` crate for GPU bandwidth (no hard-coded numbers).

This keeps the analysis single-sourced in Rust and consistent with every other tool; "extending the
matcher" stays a torch-concrete edit (add an `op_type` case), per the N9/N12 discipline.

## Alternatives considered

- **Python matcher over `torch.fx.Graph` (rejected).** Most natural against the live graph, but it
  splits graph analysis across two languages, makes Tool 6 the only tool depending on live Python
  graph state rather than the serialized contract, and duplicates the FX-consumption path that
  `cls-wl-diff` already owns in Rust. The only thing it would buy — direct access to the live graph —
  is unnecessary, because `FxNode` already carries op type + edges + attrs.
- **Hybrid: Python candidate-finder → Rust renderer (rejected).** Strictly more moving parts than
  either pure option, for no benefit here.

## Consequences

- Tool 6 lands as `cls-analyzer::fusion` (matcher + foldability + cost model + renderer) + a
  `cl fusion-detect` CLI subcommand, mirroring how `roofline` / `lint` are structured.
- **Shape metadata for the cost model**: the matcher needs only `op_type` + edges; the cost model
  (a later change) needs tensor shapes (M/N/K0/K1/dtype). If `FxNode.attrs` does not already carry
  them, the *Tool 2 collector* (Python) is extended to dump them — the shape capture stays on the
  collection side, the analysis stays in Rust. (Tracked for the cost-model section, not this one.)
- If a future pattern genuinely needs live-graph information the serialized form can't represent,
  this ADR is revisited; nothing in Pattern A does.
