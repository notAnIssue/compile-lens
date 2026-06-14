# ADR-038: Capture FX node output shapes so the fusion cost model can run end-to-end

- **Status**: Accepted
- **Date**: 2026-06-14
- **Deciders**: project maintainer
- **Related**: ADR-037 (Tool 6 is a Rust analyzer over the serialized FX graph — its Consequences pre-committed this: "if `FxNode.attrs` does not already carry [shapes], the Tool 2 collector is extended to dump them"); ADR-024 (the `FxNode` contract); ADR-006 (Python collects → `.cls.json`; Rust analyzes).

## Context

Tool 6's matcher finds `GEMM-Residual-RMSNorm-GEMM` (Pattern A) opportunities from `op_type` + edges
alone, and its analytical cost model (the HBM-traffic roofline) was implemented and unit-tested. But
the two were not connected end-to-end: `analyze()` emitted each `FusionOpportunity` with its location,
confidence, and suggested kernel, but with `shape` / `baseline_hbm_bytes` / `fused_hbm_bytes` /
`estimated_speedup` left `None` — so the cost model was dead code on a real session.

The reason is a gap in the serialized graph. The FX collector (`_serialize_node`) was built for the
WL structural diff (Tool 2a), which needs only op types, edges, and scalar attrs — never tensor
shapes. So it never read `node.meta['val']`, where the shapes live. Tool 6 is the first consumer that
needs shapes (M/N/K0/K1/dtype to size HBM traffic). The collector's contract was right-sized for its
consumers at the time; Tool 6 is a new consumer with a new need.

A fusion detector that reports "there's an opportunity here" without "and it would move ~2× less HBM
traffic" is a husk: the speedup estimate is the evidence that makes a suggestion actionable and lets
opportunities be ranked. Completing this is what makes Tool 6 useful, not just structurally correct.

## Decision

**Extend the FX collector to capture each node's concrete output shape and dtype, as two new typed
`FxNode` fields, and wire `analyze()` to derive the GEMM dimensions and call the cost model.**

1. **Collector** (`_serialize_node`): read `node.meta['val']` and record `out_shape` (a list of
   `int`) and `out_dtype` (the torch dtype name, `torch.` prefix stripped, e.g. `bfloat16`).
   `aot_autograd` traces with fake tensors, so each aten node's `meta['val']` is the `FakeTensor`
   it would produce — carrying `.shape` and `.dtype`.
2. **Schema**: `FxNode` gains typed, optional `out_shape: [int]` and `out_dtype: string` (Rust +
   Python + JSON-Schema mirrors). Additive and skip-when-empty — an artifact written before this
   field existed simply omits it, and structure-only consumers (the WL-diff) ignore it.
3. **Analyzer**: for each match, derive `M, N` from GEMM1's output shape, `K1` from GEMM2's output,
   `K0` from GEMM1's `[M, K0]` activation input (the input whose leading-dim product equals `M`),
   and the dtype from GEMM1's output; call the cost model; fill the record.

**Shapes go in typed fields, not the `attrs` bag.** A shape is first-class, structured, and reusable
(the roofline tool can use it too), so it deserves a typed field that deserializes cleanly to
`Vec<u64>` and is self-documenting — unlike `attrs`, which is the grab-bag for one-off scalar
constants.

The cost is **best-effort, not required**: if a concrete shape wasn't captured — `meta` absent, the
value isn't a single tensor, or the shape is dynamic (symbolic `SymInt`, which has no byte size) —
the opportunity is still reported, with the cost fields left `None`. The cost stays unknown rather
than guessed, consistent with the tool's analytical-only discipline (N10).

## Alternatives considered

- **Caller injects shapes via a callback (rejected).** Would avoid touching the collector, but Tool 6
  is an offline CLI analyzer: `cl fusion-detect session.cls.json` — the session file is the entire
  input, there is no live caller to supply shapes. (This is the inverse of an in-process harness like
  Tool 5's autotuner, which *does* have a live caller; "where does the shape come from" is decided by
  "is there a live caller". Offline ⇒ the shape must be in the artifact.)
- **Ship Tool 6 without cost, structural detection only (rejected).** Zero collector change, but
  halves the value — no ranking, no speedup, the cost model left dead — which guts the point of a
  fusion *detector*.
- **Infer shapes from existing scalar attrs (rejected).** The current attrs carry only scalar
  constants (`alpha`, …); there is no shape information to infer from. Not viable.

## Consequences

- The end-to-end path is complete: on a graph whose shapes were captured, `analyze()` returns
  opportunities with shape + baseline/fused HBM bytes + estimated speedup; the cost model is live.
- **Verified on a real capture.** A `Linear → +residual → RMSNorm → Linear` block compiled with
  `torch.compile` populates `out_shape`/`out_dtype` on every tensor node (only the `output` tuple
  node has none). The capture also confirmed `nn.RMSNorm` lowers to the *decomposed*
  `pow→mean→rsqrt→mul` subgraph with overload-suffixed op names (`aten.pow.Tensor_Scalar`,
  `aten.mean.dim`, …), so the decomposed matcher is the branch that fires on real graphs — pinned by
  a Rust test built from that exact captured topology.
- **Dynamic shapes** (`torch.compile(dynamic=True)`) carry `SymInt` dims; those opportunities are
  reported without a cost rather than with a fabricated one. Concrete-shape coverage is the
  common inference case; symbolic coverage is explicitly out of scope for the analytical estimate.
- Artifact size grows by one small shape array + dtype string per tensor node — bounded and only for
  nodes that have a concrete tensor value.
