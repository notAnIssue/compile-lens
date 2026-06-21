//! Tool 6 — CODA-style algebraic fusion-opportunity detector (crown-jewel feature).
//!
//! Finds `GEMM-Residual-RMSNorm-GEMM` (Pattern A) fusion opportunities in a `torch.compile` FX
//! graph that Inductor leaves on the table — cross-module algebraic fusions its local schedule
//! rules don't perform (the `1/rms` row-constant scale can be pulled through the second GEMM and
//! applied in its epilogue, removing the full-tensor RMSNorm materialization between the two GEMMs).
//! It is **suggest-only**: it reads the serialized graph, never mutates it, and never runs a kernel.
//!
//! Three disciplines bound the tool:
//!   * **forward-only (N5)** — backward subgraphs are never analyzed: the op matchers name only
//!     *forward* ops (`is_rms_norm` matches `aten.rms_norm`, never `aten.rms_norm_backward`), and a
//!     graph carrying any backward-marked op is skipped wholesale;
//!   * **torch-concrete (N9/N12)** — patterns are named matchers over concrete `aten` ops, not a
//!     generic rewrite DSL; extending the tool means adding an `op` case, not a grammar;
//!   * **analytical roofline only (N10)** — the cost model (a later change) is an HBM-bytes roofline
//!     estimate, never a measured runtime.
//!
//! Per ADR-037 this is a Rust analyzer over the serialized `FxNode[]` (ADR-024), like `lint` and
//! `wl-diff` — no live Python graph, no re-trace. Topology is read from the edges: a node's
//! `inputs` are the ids it consumes, so a node is *single-consumer* iff its id appears in exactly
//! one other node's `inputs` (one reverse scan).

use cls_schema::{ClsArtifact, FusionLocation, FusionOpportunity, FusionShape, FxNode};
use std::cmp::Ordering;
use std::collections::HashMap;

// ── torch-concrete op predicates (N9/N12) ──────────────────────────────────────────────────
/// True when `op` is `base` or an overload of it (`base.Tensor`, `base.default`, …) — but **not**
/// a different op that merely shares the prefix. `op_is("aten.addmm", "aten.add")` is false (the
/// next char is `m`, not `.`), which is exactly how `aten.add` is told apart from `aten.addmm`.
fn op_is(op: &str, base: &str) -> bool {
    op == base
        || op
            .strip_prefix(base)
            .is_some_and(|rest| rest.starts_with('.'))
}

fn is_matmul(op: &str) -> bool {
    op_is(op, "aten.mm")
        || op_is(op, "aten.addmm")
        || op_is(op, "aten.bmm")
        || op_is(op, "aten.matmul")
}
fn is_add(op: &str) -> bool {
    op_is(op, "aten.add")
}
fn is_mul(op: &str) -> bool {
    op_is(op, "aten.mul")
}
/// Forward RMSNorm only (N5): `aten.rms_norm[.overload]`, never `aten.rms_norm_backward`.
fn is_rms_norm(op: &str) -> bool {
    op_is(op, "aten.rms_norm")
}

// ── topology over the serialized edges ──────────────────────────────────────────────────────
/// Every node that lists `id` among its `inputs` (i.e. consumes it).
fn consumers_of<'a>(nodes: &'a [FxNode], id: &str) -> Vec<&'a FxNode> {
    nodes
        .iter()
        .filter(|node| node.inputs.iter().any(|i| i == id))
        .collect()
}

/// The sole consumer of `id`, or `None` if it has zero or more than one (the single-consumer
/// topology constraint Pattern A's safety rests on).
fn single_consumer<'a>(nodes: &'a [FxNode], id: &str) -> Option<&'a FxNode> {
    let consumers = consumers_of(nodes, id);
    (consumers.len() == 1).then(|| consumers[0])
}

// ── Pattern A ───────────────────────────────────────────────────────────────────────────────
/// A matched `GEMM-Residual-RMSNorm-GEMM` chain (node ids into the graph).
#[derive(Debug, Clone, PartialEq)]
pub struct PatternAMatch {
    pub gemm1: String,
    pub add: String,
    /// The RMSNorm node(s): one for a native `aten.rms_norm`, several for the decomposed subgraph.
    pub rms_norm_ids: Vec<String>,
    pub gemm2: String,
    /// `true` when the RMSNorm was matched as a decomposed `pow→mean→rsqrt→mul` subgraph rather
    /// than a single `aten.rms_norm` (drives a lower confidence in the report).
    pub decomposed: bool,
}

/// An RMSNorm matched between the residual `add` and the second GEMM.
struct RmsMatch {
    /// The node whose output feeds the second GEMM (the `aten.rms_norm`, or the final decomposed
    /// `mul`).
    output_id: String,
    node_ids: Vec<String>,
    decomposed: bool,
}

/// Find every Pattern A opportunity in one FX graph.
///
/// For each first GEMM whose sole consumer is a residual `add`, match an RMSNorm (native or
/// decomposed) on the add's output, then require that RMSNorm's sole consumer be the second GEMM.
/// The single-consumer links are what make the fusion safe to suggest — a tensor that escapes
/// elsewhere can't be folded away.
pub fn find_pattern_a(nodes: &[FxNode]) -> Vec<PatternAMatch> {
    // N5 forward-only guard: a backward graph carries backward-marked ops; never analyze it.
    if nodes.iter().any(|node| node.op_type.contains("backward")) {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for gemm1 in nodes.iter().filter(|node| is_matmul(&node.op_type)) {
        let Some(add) = single_consumer(nodes, &gemm1.id) else {
            continue;
        };
        if !is_add(&add.op_type) {
            continue;
        }
        let Some(rms) = match_rms_norm(nodes, add) else {
            continue;
        };
        let Some(gemm2) = single_consumer(nodes, &rms.output_id) else {
            continue;
        };
        if !is_matmul(&gemm2.op_type) {
            continue;
        }
        matches.push(PatternAMatch {
            gemm1: gemm1.id.clone(),
            add: add.id.clone(),
            rms_norm_ids: rms.node_ids,
            gemm2: gemm2.id.clone(),
            decomposed: rms.decomposed,
        });
    }
    matches
}

/// Match an RMSNorm on `add`'s output: a single native `aten.rms_norm`, else the decomposed
/// subgraph.
fn match_rms_norm(nodes: &[FxNode], add: &FxNode) -> Option<RmsMatch> {
    let consumers = consumers_of(nodes, &add.id);

    // Native: the add output's one consumer is an `aten.rms_norm`.
    if consumers.len() == 1 && is_rms_norm(&consumers[0].op_type) {
        return Some(RmsMatch {
            output_id: consumers[0].id.clone(),
            node_ids: vec![consumers[0].id.clone()],
            decomposed: false,
        });
    }

    match_decomposed_rms_norm(nodes, &consumers)
}

/// Match a decomposed RMSNorm `x * rsqrt(mean(x²)[+eps]) [* weight]`, where `x` is the residual
/// add's output. In this form `x` feeds *two* nodes — `pow` and the `x*rsqrt` mul — so the
/// single-consumer rule becomes "the add output's consumers are exactly the RMSNorm's entry nodes,
/// nothing escapes."
fn match_decomposed_rms_norm(nodes: &[FxNode], add_consumers: &[&FxNode]) -> Option<RmsMatch> {
    if add_consumers.len() != 2 {
        return None; // x escapes to something other than the RMSNorm entry (or isn't this shape)
    }
    let pow = add_consumers
        .iter()
        .copied()
        .find(|n| op_is(&n.op_type, "aten.pow"))?;
    let mul1 = add_consumers.iter().copied().find(|n| is_mul(&n.op_type))?;

    // pow → mean → [add eps] → rsqrt, and that rsqrt must feed mul1.
    let mean = single_consumer(nodes, &pow.id).filter(|n| op_is(&n.op_type, "aten.mean"))?;
    let after_mean = single_consumer(nodes, &mean.id)?;
    let (eps_add, rsqrt) = if is_add(&after_mean.op_type) {
        let rsqrt =
            single_consumer(nodes, &after_mean.id).filter(|n| op_is(&n.op_type, "aten.rsqrt"))?;
        (Some(after_mean), rsqrt)
    } else if op_is(&after_mean.op_type, "aten.rsqrt") {
        (None, after_mean)
    } else {
        return None;
    };
    if !mul1.inputs.iter().any(|i| i == &rsqrt.id) {
        return None; // the rsqrt doesn't actually feed the x*rsqrt mul
    }

    let mut node_ids = vec![pow.id.clone(), mean.id.clone()];
    if let Some(eps) = eps_add {
        node_ids.push(eps.id.clone());
    }
    node_ids.push(rsqrt.id.clone());
    node_ids.push(mul1.id.clone());

    // Optional weight mul: if mul1's sole consumer is another mul, that's the RMSNorm output;
    // otherwise mul1 itself is the output (no learned weight).
    let output = match single_consumer(nodes, &mul1.id) {
        Some(weight_mul) if is_mul(&weight_mul.op_type) => {
            node_ids.push(weight_mul.id.clone());
            weight_mul
        }
        _ => mul1,
    };

    Some(RmsMatch {
        output_id: output.id.clone(),
        node_ids,
        decomposed: true,
    })
}

/// Scan a session's compiled graphs for Pattern A fusion opportunities. Fills the pattern, the FX
/// node range, the suggested kernel, and the confidence; and — when the collector captured concrete
/// shapes (ADR-038) — the GEMM shape and the analytical HBM-traffic estimate via the cost model.
///
/// The cost is best-effort, not required: an opportunity whose shapes weren't captured (meta absent
/// or a dynamic/symbolic shape) is still reported, just with `shape`/bytes/`estimated_speedup` left
/// `None` — the structural finding stands on its own, and the cost stays unknown rather than guessed
/// (N10).
pub fn analyze(artifact: &ClsArtifact) -> Vec<FusionOpportunity> {
    let mut out = Vec::new();
    for graph in &artifact.compiled_graphs {
        let by_id: HashMap<&str, &FxNode> = graph
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect();
        for m in find_pattern_a(&graph.nodes) {
            // Derive shapes (and cost) *before* moving the match's ids into the location below.
            let cost = derive_shape(&by_id, &m).map(|(shape, dtype)| {
                let db = dtype_bytes(&dtype);
                let baseline = baseline_hbm_bytes(&shape, db);
                let fused = coda_hbm_bytes(&shape, db, DEFAULT_TILE_N);
                let speedup = estimated_speedup(baseline, fused);
                (shape, dtype, baseline, fused, speedup)
            });

            let mut fx_node_ids = vec![m.gemm1, m.add];
            fx_node_ids.extend(m.rms_norm_ids);
            fx_node_ids.push(m.gemm2);

            let (shape, baseline_hbm_bytes, fused_hbm_bytes, estimated_speedup) = match cost {
                Some((s, dtype, baseline, fused, speedup)) => (
                    Some(FusionShape {
                        m: Some(s.m),
                        n: Some(s.n),
                        k0: Some(s.k0),
                        k1: Some(s.k1),
                        dtype: Some(dtype),
                    }),
                    Some(baseline as f64),
                    Some(fused as f64),
                    Some(speedup),
                ),
                None => (None, None, None, None),
            };

            out.push(FusionOpportunity {
                pattern_id: "A".to_string(),
                location: Some(FusionLocation {
                    fx_node_ids,
                    src_lineno_range: None,
                }),
                shape,
                baseline_hbm_bytes,
                fused_hbm_bytes,
                estimated_speedup,
                suggested_fusion: Some(
                    "fold per-row 1/rms into the second GEMM epilogue".to_string(),
                ),
                // Native exact match is high-confidence; a decomposed subgraph match is medium.
                confidence: Some(if m.decomposed { "medium" } else { "high" }.to_string()),
            });
        }
    }
    out
}

/// Derive the GEMM dimensions (and dtype) a Pattern A match spans, from the captured node output
/// shapes. `None` whenever a needed shape wasn't captured — the caller then leaves the cost unset.
///
/// Reading the dims off the two matched GEMMs and GEMM1's activation input:
/// GEMM1 output is `[…, M, N]`, GEMM2 output is `[…, M, K1]`, and `K0` is GEMM1's contraction dim —
/// the trailing dim of its `[…, M, K0]` activation operand, found as the **first** input whose
/// leading-dim product equals `M`. The `[N]` bias never matches; the weight `[K0, N]` matches only
/// in a **square contraction** (`K0 == M`), and there disambiguation relies on the activation
/// preceding the weight in the operand order — the convention torch's `mm`/`addmm` emit (the
/// activation is the first matrix operand). The dtype is taken from GEMM1's output.
fn derive_shape(by_id: &HashMap<&str, &FxNode>, m: &PatternAMatch) -> Option<(Shape, String)> {
    let g1 = by_id.get(m.gemm1.as_str())?;
    let g2 = by_id.get(m.gemm2.as_str())?;
    let (rows, n) = rows_cols(&g1.out_shape)?;
    let (_g2_rows, k1) = rows_cols(&g2.out_shape)?;
    let k0 = g1
        .inputs
        .iter()
        .filter_map(|id| by_id.get(id.as_str()))
        .find_map(|inp| {
            rows_cols(&inp.out_shape)
                .filter(|&(r, _)| r == rows)
                .map(|(_, c)| c)
        })?;
    let dtype = g1.out_dtype.clone()?;
    Some((Shape { m: rows, n, k0, k1 }, dtype))
}

/// `(product of leading dims, last dim)` for a rank-≥2 shape — the GEMM's row count (batch dims
/// folded into `M`) and its column count. `None` for a 0-D/1-D shape (not a GEMM output).
fn rows_cols(shape: &[u64]) -> Option<(u64, u64)> {
    if shape.len() < 2 {
        return None;
    }
    let (&cols, leading) = shape.split_last().unwrap();
    Some((leading.iter().product(), cols))
}

// ── analytical HBM-traffic cost model (N10) ─────────────────────────────────────────────────
//
// A memory-bound roofline: total HBM bytes moved, baseline (four separate kernels) vs CODA-fused
// (GEMM1+residual+partial-RMS in one kernel, an aux reduction, then GEMM2 applying the per-row
// reciprocal-rms in its epilogue — the RMSNorm never materializes). The speedup is the byte ratio
// under the memory-bound assumption: an **upper bound used to rank opportunities, not a promised
// number** — real GEMMs are often compute-bound, so the achieved speedup is lower (the CODA paper
// measures ~1.05–1.15x on LLaMA shapes). Bandwidth cancels in the ratio, so it is not needed here.

/// The GEMM shapes a Pattern A opportunity spans: `RMSNorm(x·W0[M×K0→M×N] + z)·W1[N×K1→M×K1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub m: u64,
    pub n: u64,
    pub k0: u64,
    pub k1: u64,
}

/// Default autotune tile width along N for the partial-statistics accounting (CODA uses 128).
pub const DEFAULT_TILE_N: u64 = 128;

/// Bytes per element for a dtype name. Accepts both the short forms (`bf16`, `fp32`) and the torch
/// names the collector emits (`bfloat16`, `float32`, `float8_e4m3fn`). Inference activations default
/// to 2 (bf16 / fp16).
pub fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "fp64" | "float64" => 8,
        "fp32" | "float32" => 4,
        "fp16" | "float16" | "bf16" | "bfloat16" => 2,
        d if d.starts_with("float8") || d.starts_with("fp8") || d == "e4m3" || d == "e5m2" => 1,
        _ => 2,
    }
}

/// HBM bytes for the standard path: GEMM1, residual add, RMSNorm, GEMM2 as four separate kernels.
/// `gemm_fixed` (weight reads + final output) is identical in both paths; the activations stream
/// through HBM eight times (GEMM1 writes D; the add reads D, reads the residual, writes D'; RMSNorm
/// reads D' twice and writes Y; GEMM2 reads Y).
pub fn baseline_hbm_bytes(shape: &Shape, dtype_bytes: u64) -> u64 {
    let gemm_fixed = gemm_fixed_elems(shape);
    let activations = 8 * shape.m * shape.n;
    (gemm_fixed + activations) * dtype_bytes
}

/// HBM bytes for the CODA-fused path. The activations stream through HBM three times (kernel A
/// writes D' and reads the residual; kernel C reads D') — the RMSNorm is never materialized — plus a
/// little fp32 traffic for the partial statistics and the per-row reciprocal-rms.
pub fn coda_hbm_bytes(shape: &Shape, dtype_bytes: u64, tile_n: u64) -> u64 {
    let gemm_fixed = gemm_fixed_elems(shape);
    let activations = 3 * shape.m * shape.n;
    // div_ceil, not floor: a final partial tile (n not a multiple of tile_n) still writes its own
    // partial statistics, so it counts toward the fp32 traffic below.
    let tiles = shape.n.div_ceil(tile_n.max(1)).max(1);
    // fp32 (4 bytes): partial stats written by A and read by aux; r written by aux and read by C.
    let fp32 = 4 * (2 * shape.m * tiles + 2 * shape.m);
    (gemm_fixed + activations) * dtype_bytes + fp32
}

/// Weight reads and final output (in elements) — unchanged by the fusion, so it appears in both.
fn gemm_fixed_elems(shape: &Shape) -> u64 {
    shape.m * shape.k0 + shape.k0 * shape.n + shape.n * shape.k1 + shape.m * shape.k1
}

/// Memory-bound speedup estimate: the HBM-traffic ratio. An upper bound for ranking, not a promise.
pub fn estimated_speedup(baseline_bytes: u64, coda_bytes: u64) -> f64 {
    if coda_bytes == 0 {
        return 0.0;
    }
    baseline_bytes as f64 / coda_bytes as f64
}

// ── report rendering ──────────────────────────────────────────────────────────────────────────
//
// Renders the detector's opportunities, highest estimated speedup first. `Json` is schema-aligned
// (`{ "fusion_opportunities": [...] }` — the artifact's own field, so the report round-trips back
// into the schema type); `Markdown` follows the worked example in the design note. An opportunity
// whose shapes weren't captured still renders — with its location, confidence, and suggested kernel
// — but with an honest "not estimated" line instead of a fabricated cost.

/// Output format for the fusion report (mirrors the other tools' two shapes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Markdown,
    Json,
}

/// A serialize-only view of the report under the schema's `fusion_opportunities` key, borrowing the
/// opportunities so rendering never clones the artifact's records.
#[derive(serde::Serialize)]
struct FusionReport<'a> {
    fusion_opportunities: Vec<&'a FusionOpportunity>,
}

/// Render fusion opportunities, ranked by estimated speedup (descending; opportunities with no
/// estimate sort last, then by `pattern_id` for a stable order).
pub fn render(opportunities: &[FusionOpportunity], format: Format) -> String {
    let mut ranked: Vec<&FusionOpportunity> = opportunities.iter().collect();
    ranked.sort_by(|a, b| cmp_speedup_desc(a, b).then_with(|| a.pattern_id.cmp(&b.pattern_id)));
    match format {
        Format::Json => {
            let report = FusionReport {
                fusion_opportunities: ranked,
            };
            serde_json::to_string_pretty(&report).expect("FusionReport serializes")
        }
        Format::Markdown => render_markdown(&ranked),
    }
}

/// Select the opportunities to report: drop those whose estimated speedup is below `min_speedup`
/// (an opportunity with *no* estimate is kept — it can't be ranked, but it is still a real finding),
/// rank by speedup descending, and keep the top `top_k`. This is the detector's report policy; the
/// CLI just supplies the thresholds.
pub fn select(
    opportunities: Vec<FusionOpportunity>,
    min_speedup: f64,
    top_k: usize,
) -> Vec<FusionOpportunity> {
    let mut kept: Vec<FusionOpportunity> = opportunities
        .into_iter()
        // `map_or(true, …)` (not `is_none_or`, which is >MSRV): no estimate -> keep it.
        .filter(|o| o.estimated_speedup.map_or(true, |s| s >= min_speedup))
        .collect();
    kept.sort_by(|a, b| cmp_speedup_desc(a, b).then_with(|| a.pattern_id.cmp(&b.pattern_id)));
    kept.truncate(top_k);
    kept
}

/// Speedup descending; an opportunity carrying an estimate ranks ahead of one that doesn't.
fn cmp_speedup_desc(a: &FusionOpportunity, b: &FusionOpportunity) -> Ordering {
    match (a.estimated_speedup, b.estimated_speedup) {
        (Some(x), Some(y)) => y.partial_cmp(&x).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// The human-readable name for a matched pattern id.
fn pattern_name(pattern_id: &str) -> &str {
    match pattern_id {
        "A" => "GEMM-Residual-RMSNorm-GEMM",
        _ => "fusion pattern",
    }
}

/// Format bytes as megabytes (MB = 10⁶ B). Whole MB at model scale (matching the design note's
/// worked example, e.g. `1757 MB`); two decimals below 1 MB so a small graph reads `0.21 MB`
/// rather than a misleading `0 MB`.
fn fmt_mb(bytes: f64) -> String {
    let mb = bytes / 1_000_000.0;
    if mb >= 1.0 {
        format!("{} MB", mb.round() as u64)
    } else {
        format!("{mb:.2} MB")
    }
}

fn render_markdown(ranked: &[&FusionOpportunity]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "## Fusion Opportunities Detected");

    if ranked.is_empty() {
        let _ = writeln!(out, "\nNo fusion opportunities detected.");
        return out;
    }

    for (i, opp) in ranked.iter().enumerate() {
        let name = pattern_name(&opp.pattern_id);
        let _ = writeln!(out, "\n### #{}: {name}", i + 1);
        let _ = writeln!(out, "- Pattern: {} ({name})", opp.pattern_id);

        if let Some(loc) = &opp.location {
            let nodes = loc.fx_node_ids.join(", ");
            match &loc.src_lineno_range {
                Some(src) => {
                    let _ = writeln!(out, "- Location: FX nodes {nodes}, src: {src}");
                }
                None => {
                    let _ = writeln!(out, "- Location: FX nodes {nodes}");
                }
            }
        }

        if let Some(shape) = &opp.shape {
            if let (Some(m), Some(n), Some(k0), Some(k1)) = (shape.m, shape.n, shape.k0, shape.k1) {
                let dtype = shape.dtype.as_deref().unwrap_or("?");
                let _ = writeln!(
                    out,
                    "- Shape: M={m}, K0={k0}, N={n}, K1={k1}, dtype={dtype}"
                );
            }
        }

        match (
            opp.baseline_hbm_bytes,
            opp.fused_hbm_bytes,
            opp.estimated_speedup,
        ) {
            (Some(baseline), Some(fused), Some(speedup)) => {
                let _ = writeln!(out, "- Baseline HBM traffic: {}", fmt_mb(baseline));
                let _ = writeln!(out, "- CODA-fused HBM traffic: {}", fmt_mb(fused));
                let _ = writeln!(
                    out,
                    "- Estimated speedup: ~{speedup:.2}× (memory-bound upper bound — ranks \
                     opportunities, not a promise; real is lower, ~1.05–1.15× per the paper, \
                     GEMMs being compute-bound)"
                );
            }
            _ => {
                let _ = writeln!(
                    out,
                    "- HBM traffic: not estimated (tensor shapes unavailable for this graph)"
                );
            }
        }

        if let Some(fusion) = &opp.suggested_fusion {
            let _ = writeln!(out, "- Suggested fusion: {fusion}");
        }
        if let Some(confidence) = &opp.confidence {
            let _ = writeln!(out, "- Confidence: {confidence}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: &str, op: &str, inputs: &[&str]) -> FxNode {
        FxNode {
            id: id.to_string(),
            op_type: op.to_string(),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: Default::default(),
            out_shape: Vec::new(),
            out_dtype: None,
        }
    }

    /// g1(addmm) → add(residual) → rms_norm → g2(mm), a clean single-consumer chain.
    fn native_pattern_a() -> Vec<FxNode> {
        vec![
            n("g1", "aten.addmm", &["x", "w0", "bias"]),
            n("add", "aten.add", &["g1", "resid"]),
            n("rms", "aten.rms_norm", &["add", "gamma"]),
            n("g2", "aten.mm", &["rms", "w1"]),
        ]
    }

    #[test]
    fn derive_shape_square_contraction_picks_activation_by_operand_order() {
        // Square contraction M == K0 == 512: the weight [K0, N] = [512, 256] also has leading dim
        // == M, so it matches the row filter too. derive_shape must still report K0 = 512 (the
        // activation's trailing dim), not the weight's N = 256 — it relies on the activation
        // preceding the weight in g1.inputs (torch mm/addmm operand order).
        let mut x = n("x", "placeholder", &[]);
        x.out_shape = vec![512, 512]; // activation [M, K0]
        let mut w0 = n("w0", "placeholder", &[]);
        w0.out_shape = vec![512, 256]; // weight [K0, N] — rows == M in the square case
        let mut bias = n("bias", "placeholder", &[]);
        bias.out_shape = vec![256];
        let mut g1 = n("g1", "aten.addmm", &["x", "w0", "bias"]); // activation first
        g1.out_shape = vec![512, 256]; // [M, N]
        g1.out_dtype = Some("bfloat16".to_string());
        let mut g2 = n("g2", "aten.mm", &["rms", "w1"]);
        g2.out_shape = vec![512, 128]; // [M, K1]
        let nodes = [x, w0, bias, g1, g2];
        let by_id: HashMap<&str, &FxNode> = nodes.iter().map(|nd| (nd.id.as_str(), nd)).collect();
        let m = PatternAMatch {
            gemm1: "g1".into(),
            add: "add".into(),
            rms_norm_ids: vec!["rms".into()],
            gemm2: "g2".into(),
            decomposed: false,
        };

        let (shape, dtype) = derive_shape(&by_id, &m).expect("shape derives");
        assert_eq!((shape.m, shape.n, shape.k1), (512, 256, 128));
        assert_eq!(
            shape.k0, 512,
            "must pick activation K0 (512), not weight N (256)"
        );
        assert_eq!(dtype, "bfloat16");
    }

    #[test]
    fn coda_counts_a_partial_final_tile() {
        // n = 130 over tile_n = 128 is two tiles (one full + one partial); the partial tile still
        // writes its own partial statistics. Floor division would miscount it as one tile.
        let shape = Shape {
            m: 8,
            n: 130,
            k0: 64,
            k1: 64,
        };
        let two_tiles = coda_hbm_bytes(&shape, 2, 128); // div_ceil(130, 128) = 2
        let one_tile = coda_hbm_bytes(&shape, 2, 130); // div_ceil(130, 130) = 1
        assert!(
            two_tiles > one_tile,
            "the partial 2nd tile must add fp32 stat traffic"
        );
        // The only difference is one tile's worth of fp32 partial-stats: 4 bytes * 2 * m.
        assert_eq!(two_tiles - one_tile, 4 * 2 * shape.m);
    }

    #[test]
    fn op_is_distinguishes_add_from_addmm() {
        assert!(is_add("aten.add"));
        assert!(is_add("aten.add.Tensor"));
        assert!(!is_add("aten.addmm"));
        assert!(is_matmul("aten.addmm"));
        assert!(is_rms_norm("aten.rms_norm"));
        assert!(!is_rms_norm("aten.rms_norm_backward"));
    }

    #[test]
    fn test_pattern_a_exact_match_addmm_native_rms_norm() {
        let m = find_pattern_a(&native_pattern_a());
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].gemm1, "g1");
        assert_eq!(m[0].add, "add");
        assert_eq!(m[0].rms_norm_ids, vec!["rms"]);
        assert_eq!(m[0].gemm2, "g2");
        assert!(!m[0].decomposed);
    }

    #[test]
    fn test_pattern_a_match_with_mm_no_bias() {
        let mut g = native_pattern_a();
        g[0] = n("g1", "aten.mm", &["x", "w0"]); // first GEMM without a bias
        let m = find_pattern_a(&g);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].gemm1, "g1");
    }

    #[test]
    fn test_pattern_a_match_with_decomposed_rms_norm() {
        let g = vec![
            n("g1", "aten.mm", &["x", "w0"]),
            n("add", "aten.add", &["g1", "resid"]),
            n("pow", "aten.pow", &["add"]),
            n("mean", "aten.mean", &["pow"]),
            n("eps", "aten.add", &["mean"]),
            n("rsqrt", "aten.rsqrt", &["eps"]),
            n("mul1", "aten.mul", &["add", "rsqrt"]),
            n("mul2", "aten.mul", &["mul1", "gamma"]),
            n("g2", "aten.mm", &["mul2", "w1"]),
        ];
        let m = find_pattern_a(&g);
        assert_eq!(m.len(), 1);
        assert!(m[0].decomposed);
        assert_eq!(
            m[0].rms_norm_ids,
            vec!["pow", "mean", "eps", "rsqrt", "mul1", "mul2"]
        );
        assert_eq!(m[0].gemm2, "g2");
    }

    #[test]
    fn test_pattern_a_match_decomposed_without_eps_or_weight() {
        // Minimal decomposition: no eps add, no learned-weight mul.
        let g = vec![
            n("g1", "aten.mm", &["x", "w0"]),
            n("add", "aten.add", &["g1", "resid"]),
            n("pow", "aten.pow", &["add"]),
            n("mean", "aten.mean", &["pow"]),
            n("rsqrt", "aten.rsqrt", &["mean"]),
            n("mul1", "aten.mul", &["add", "rsqrt"]),
            n("g2", "aten.mm", &["mul1", "w1"]),
        ];
        let m = find_pattern_a(&g);
        assert_eq!(m.len(), 1);
        assert!(m[0].decomposed);
        assert_eq!(m[0].rms_norm_ids, vec!["pow", "mean", "rsqrt", "mul1"]);
    }

    #[test]
    fn test_pattern_a_reject_when_residual_has_multiple_consumers() {
        let mut g = native_pattern_a();
        g.push(n("leak", "aten.relu", &["add"])); // the residual output also escapes to relu
        assert!(find_pattern_a(&g).is_empty());
    }

    #[test]
    fn test_pattern_a_reject_when_rms_norm_has_multiple_consumers() {
        let mut g = native_pattern_a();
        g.push(n("leak", "aten.relu", &["rms"])); // the RMSNorm output also escapes to relu
        assert!(find_pattern_a(&g).is_empty());
    }

    #[test]
    fn test_pattern_a_reject_backward_subgraph() {
        // N5 forward-only: a backward-marked op makes the whole graph off-limits.
        let g = vec![
            n("g1", "aten.mm", &["x", "w0"]),
            n("add", "aten.add", &["g1", "resid"]),
            n("rmsb", "aten.rms_norm_backward", &["add", "gamma"]),
            n("g2", "aten.mm", &["rmsb", "w1"]),
        ];
        assert!(find_pattern_a(&g).is_empty());
    }

    #[test]
    fn test_pattern_a_no_false_positive_on_non_rms_chain() {
        // GEMM → residual → activation → GEMM is *not* Pattern A (no RMSNorm between the GEMMs).
        let g = vec![
            n("g1", "aten.mm", &["x", "w0"]),
            n("add", "aten.add", &["g1", "resid"]),
            n("act", "aten.relu", &["add"]),
            n("g2", "aten.mm", &["act", "w1"]),
        ];
        assert!(find_pattern_a(&g).is_empty());
    }

    #[test]
    fn test_pattern_a_detect_all_three_locations() {
        let mut g = Vec::new();
        for i in 0..3 {
            g.push(n(&format!("g1_{i}"), "aten.addmm", &["x", "w0", "bias"]));
            g.push(n(
                &format!("add_{i}"),
                "aten.add",
                &[&format!("g1_{i}"), "resid"],
            ));
            g.push(n(
                &format!("rms_{i}"),
                "aten.rms_norm",
                &[&format!("add_{i}"), "gamma"],
            ));
            g.push(n(
                &format!("g2_{i}"),
                "aten.mm",
                &[&format!("rms_{i}"), "w1"],
            ));
        }
        assert_eq!(find_pattern_a(&g).len(), 3);
    }

    #[test]
    fn analyze_emits_fusion_opportunity_from_a_compiled_graph() {
        let json = r#"{
            "schema_version":"0.5.0",
            "session":{"id":"00000000-0000-4000-8000-000000000000","timestamp":"t",
                       "torch_version":"2.6.0","redaction_policy":"default-strict"},
            "compiled_graphs":[{"graph_id":"g","nodes":[
                {"id":"g1","op_type":"aten.addmm","inputs":["x","w0","bias"]},
                {"id":"add","op_type":"aten.add","inputs":["g1","resid"]},
                {"id":"rms","op_type":"aten.rms_norm","inputs":["add","gamma"]},
                {"id":"g2","op_type":"aten.mm","inputs":["rms","w1"]}
            ]}]
        }"#;
        let artifact: ClsArtifact = serde_json::from_str(json).expect("artifact parses");
        let found = analyze(&artifact);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].pattern_id, "A");
        assert_eq!(found[0].confidence.as_deref(), Some("high"));
        assert_eq!(
            found[0].suggested_fusion.as_deref(),
            Some("fold per-row 1/rms into the second GEMM epilogue")
        );
        let ids = &found[0].location.as_ref().unwrap().fx_node_ids;
        assert_eq!(ids, &vec!["g1", "add", "rms", "g2"]);
        // No shapes captured -> the structural finding stands, but the cost stays unset (N10).
        assert!(found[0].shape.is_none(), "no shapes -> no shape record");
        assert!(found[0].baseline_hbm_bytes.is_none());
        assert!(found[0].estimated_speedup.is_none());
    }

    #[test]
    fn analyze_fills_cost_when_shapes_are_captured() {
        // The same canonical LLaMA shape as `cost_model_on_the_canonical_llama_shape` (M=8192,
        // N=11008, K0=4096, K1=4096, bf16), now driven end-to-end from captured node out_shapes —
        // so the matcher, shape derivation, and cost model agree on the worked-example numbers.
        let json = r#"{
            "schema_version":"0.5.0",
            "session":{"id":"00000000-0000-4000-8000-000000000000","timestamp":"t",
                       "torch_version":"2.6.0","redaction_policy":"default-strict"},
            "compiled_graphs":[{"graph_id":"g","nodes":[
                {"id":"x","op_type":"placeholder","out_shape":[8192,4096],"out_dtype":"bfloat16"},
                {"id":"w0","op_type":"placeholder","out_shape":[4096,11008],"out_dtype":"bfloat16"},
                {"id":"bias","op_type":"placeholder","out_shape":[11008],"out_dtype":"bfloat16"},
                {"id":"resid","op_type":"placeholder","out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"gamma","op_type":"placeholder","out_shape":[11008],"out_dtype":"bfloat16"},
                {"id":"w1","op_type":"placeholder","out_shape":[11008,4096],"out_dtype":"bfloat16"},
                {"id":"g1","op_type":"aten.addmm","inputs":["bias","x","w0"],"out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"add","op_type":"aten.add","inputs":["g1","resid"],"out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"rms","op_type":"aten.rms_norm","inputs":["add","gamma"],"out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"g2","op_type":"aten.mm","inputs":["rms","w1"],"out_shape":[8192,4096],"out_dtype":"bfloat16"}
            ]}]
        }"#;
        let artifact: ClsArtifact = serde_json::from_str(json).expect("artifact parses");
        let found = analyze(&artifact);
        assert_eq!(found.len(), 1);

        let shape = found[0]
            .shape
            .as_ref()
            .expect("shape derived from captured out_shapes");
        assert_eq!(shape.m, Some(8192));
        assert_eq!(shape.n, Some(11008));
        assert_eq!(shape.k0, Some(4096));
        assert_eq!(shape.k1, Some(4096));
        assert_eq!(shape.dtype.as_deref(), Some("bfloat16"));

        assert_eq!(found[0].baseline_hbm_bytes, Some(1_757_413_376.0));
        assert_eq!(found[0].fused_hbm_bytes, Some(861_339_648.0));
        let speedup = found[0].estimated_speedup.expect("speedup filled");
        assert!(
            (speedup - 2.04).abs() < 0.01,
            "memory-bound ~2.04x, got {speedup}"
        );
    }

    #[test]
    fn analyze_leaves_cost_unset_on_partial_shapes() {
        // GEMM2's shape is missing -> K1 (and the whole cost) can't be derived; the opportunity is
        // still reported, just without a cost (graceful, not an error).
        let json = r#"{
            "schema_version":"0.5.0",
            "session":{"id":"00000000-0000-4000-8000-000000000000","timestamp":"t",
                       "torch_version":"2.6.0","redaction_policy":"default-strict"},
            "compiled_graphs":[{"graph_id":"g","nodes":[
                {"id":"x","op_type":"placeholder","out_shape":[8192,4096],"out_dtype":"bfloat16"},
                {"id":"w0","op_type":"placeholder","out_shape":[4096,11008],"out_dtype":"bfloat16"},
                {"id":"g1","op_type":"aten.mm","inputs":["x","w0"],"out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"add","op_type":"aten.add","inputs":["g1","resid"],"out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"rms","op_type":"aten.rms_norm","inputs":["add","gamma"],"out_shape":[8192,11008],"out_dtype":"bfloat16"},
                {"id":"g2","op_type":"aten.mm","inputs":["rms","w1"]}
            ]}]
        }"#;
        let artifact: ClsArtifact = serde_json::from_str(json).expect("artifact parses");
        let found = analyze(&artifact);
        assert_eq!(found.len(), 1);
        assert!(found[0].shape.is_none(), "GEMM2 shape missing -> no cost");
        assert!(found[0].estimated_speedup.is_none());
    }

    #[test]
    fn analyze_on_a_real_captured_decomposed_graph() {
        // The exact aten graph a real `torch.compile` capture of `Linear -> +resid -> RMSNorm ->
        // Linear` produces (verified against torch 2.12 CPU): `nn.RMSNorm` lowers to the *decomposed*
        // pow/mean/eps/rsqrt/mul/mul subgraph (not a native `aten.rms_norm`), ops carry overload
        // suffixes (`.Tensor_Scalar`, `.dim`, `.Scalar`), and the Linear weights appear as `aten.t`
        // transposes feeding the GEMMs. This pins that the matcher + shape derivation + cost all fire
        // end-to-end on a real graph, through the decomposed branch.
        let json = r#"{
            "schema_version":"0.5.0",
            "session":{"id":"00000000-0000-4000-8000-000000000000","timestamp":"t",
                       "torch_version":"2.12.0","redaction_policy":"default-strict"},
            "compiled_graphs":[{"graph_id":"g","nodes":[
                {"id":"arg1_1","op_type":"placeholder","out_shape":[128],"out_dtype":"float32"},
                {"id":"arg2_1","op_type":"placeholder","out_shape":[32,64],"out_dtype":"float32"},
                {"id":"arg3_1","op_type":"placeholder","out_shape":[32,128],"out_dtype":"float32"},
                {"id":"arg4_1","op_type":"placeholder","out_shape":[128],"out_dtype":"float32"},
                {"id":"t","op_type":"aten.t.default","inputs":["arg0_1"],"out_shape":[64,128],"out_dtype":"float32"},
                {"id":"addmm","op_type":"aten.addmm.default","inputs":["arg1_1","arg2_1","t"],"out_shape":[32,128],"out_dtype":"float32"},
                {"id":"add","op_type":"aten.add.Tensor","inputs":["addmm","arg3_1"],"out_shape":[32,128],"out_dtype":"float32"},
                {"id":"pow_1","op_type":"aten.pow.Tensor_Scalar","inputs":["add"],"out_shape":[32,128],"out_dtype":"float32"},
                {"id":"mean","op_type":"aten.mean.dim","inputs":["pow_1"],"out_shape":[32,1],"out_dtype":"float32"},
                {"id":"add_1","op_type":"aten.add.Scalar","inputs":["mean"],"out_shape":[32,1],"out_dtype":"float32"},
                {"id":"rsqrt","op_type":"aten.rsqrt.default","inputs":["add_1"],"out_shape":[32,1],"out_dtype":"float32"},
                {"id":"mul","op_type":"aten.mul.Tensor","inputs":["add","rsqrt"],"out_shape":[32,128],"out_dtype":"float32"},
                {"id":"mul_1","op_type":"aten.mul.Tensor","inputs":["mul","arg4_1"],"out_shape":[32,128],"out_dtype":"float32"},
                {"id":"t_1","op_type":"aten.t.default","inputs":["arg5_1"],"out_shape":[128,64],"out_dtype":"float32"},
                {"id":"mm","op_type":"aten.mm.default","inputs":["mul_1","t_1"],"out_shape":[32,64],"out_dtype":"float32"},
                {"id":"output","op_type":"output","inputs":["mm"]}
            ]}]
        }"#;
        let artifact: ClsArtifact = serde_json::from_str(json).expect("artifact parses");
        let found = analyze(&artifact);
        assert_eq!(found.len(), 1, "one decomposed Pattern A in the real graph");
        assert_eq!(
            found[0].confidence.as_deref(),
            Some("medium"),
            "decomposed -> medium"
        );

        let shape = found[0]
            .shape
            .as_ref()
            .expect("shape derived from the real capture");
        assert_eq!(shape.m, Some(32));
        assert_eq!(shape.n, Some(128));
        assert_eq!(shape.k0, Some(64)); // contraction dim of the first GEMM, off its [M,K0] input
        assert_eq!(shape.k1, Some(64));
        assert_eq!(shape.dtype.as_deref(), Some("float32"));
        assert!(
            found[0].estimated_speedup.unwrap() > 1.0,
            "fusion saves traffic"
        );
    }

    #[test]
    fn cost_model_on_the_canonical_llama_shape() {
        // M=8192, K0=4096, N=11008, K1=4096, bf16 — the worked example.
        let shape = Shape {
            m: 8192,
            n: 11008,
            k0: 4096,
            k1: 4096,
        };
        let db = dtype_bytes("bf16");
        assert_eq!(db, 2);
        let baseline = baseline_hbm_bytes(&shape, db);
        let coda = coda_hbm_bytes(&shape, db, DEFAULT_TILE_N);
        assert_eq!(baseline, 1_757_413_376);
        assert_eq!(coda, 861_339_648);
        assert!(coda < baseline, "fusion must move less HBM traffic");
        let speedup = estimated_speedup(baseline, coda);
        assert!(
            (speedup - 2.04).abs() < 0.01,
            "memory-bound speedup ~2.04x, got {speedup}"
        );
    }

    #[test]
    fn cost_model_scales_linearly_with_dtype_bytes() {
        // The baseline is pure elementwise traffic, so fp32 moves exactly twice the bytes of bf16.
        let shape = Shape {
            m: 1024,
            n: 1024,
            k0: 512,
            k1: 512,
        };
        let bf16 = baseline_hbm_bytes(&shape, dtype_bytes("bf16"));
        let fp32 = baseline_hbm_bytes(&shape, dtype_bytes("fp32"));
        assert_eq!(fp32, 2 * bf16);
    }

    #[test]
    fn cost_model_fusion_always_saves_and_speedup_handles_zero() {
        let shape = Shape {
            m: 4096,
            n: 4096,
            k0: 4096,
            k1: 4096,
        };
        let db = dtype_bytes("bf16");
        let baseline = baseline_hbm_bytes(&shape, db);
        let coda = coda_hbm_bytes(&shape, db, DEFAULT_TILE_N);
        assert!(estimated_speedup(baseline, coda) > 1.0);
        // Degenerate guard: never divide by zero.
        assert_eq!(estimated_speedup(100, 0), 0.0);
    }

    // ── report rendering ────────────────────────────────────────────────────────────────────

    /// The canonical worked-example opportunity (matches `cost_model_on_the_canonical_llama_shape`).
    fn canonical_opp() -> FusionOpportunity {
        FusionOpportunity {
            pattern_id: "A".to_string(),
            location: Some(FusionLocation {
                fx_node_ids: vec![
                    "g1".to_string(),
                    "add".to_string(),
                    "rms".to_string(),
                    "g2".to_string(),
                ],
                src_lineno_range: None,
            }),
            shape: Some(FusionShape {
                m: Some(8192),
                n: Some(11008),
                k0: Some(4096),
                k1: Some(4096),
                dtype: Some("bf16".to_string()),
            }),
            baseline_hbm_bytes: Some(1_757_413_376.0),
            fused_hbm_bytes: Some(861_339_648.0),
            estimated_speedup: Some(2.04),
            suggested_fusion: Some("fold per-row 1/rms into the second GEMM epilogue".to_string()),
            confidence: Some("high".to_string()),
        }
    }

    #[test]
    fn render_markdown_matches_worked_example() {
        let md = render(&[canonical_opp()], Format::Markdown);
        assert!(md.contains("## Fusion Opportunities Detected"));
        assert!(md.contains("GEMM-Residual-RMSNorm-GEMM"));
        assert!(md.contains("Location: FX nodes g1, add, rms, g2"));
        assert!(md.contains("Shape: M=8192, K0=4096, N=11008, K1=4096, dtype=bf16"));
        assert!(md.contains("Baseline HBM traffic: 1757 MB"));
        assert!(md.contains("CODA-fused HBM traffic: 861 MB"));
        assert!(md.contains("~2.04×"));
        assert!(
            md.contains("upper bound"),
            "N10 honesty caveat must be present"
        );
        assert!(md.contains("fold per-row 1/rms into the second GEMM epilogue"));
        assert!(md.contains("Confidence: high"));
    }

    #[test]
    fn render_empty_is_clean() {
        let md = render(&[], Format::Markdown);
        assert!(md.contains("No fusion opportunities detected."));
    }

    #[test]
    fn render_json_is_schema_aligned_and_round_trips() {
        let json = render(&[canonical_opp()], Format::Json);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        let arr = v["fusion_opportunities"]
            .as_array()
            .expect("opportunities under the schema key");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["pattern_id"], "A");
        assert_eq!(arr[0]["estimated_speedup"], 2.04);
        // It deserializes back into the schema type (the report is genuinely schema-aligned).
        let back: Vec<FusionOpportunity> =
            serde_json::from_value(v["fusion_opportunities"].clone()).expect("round-trips");
        assert_eq!(back[0].confidence.as_deref(), Some("high"));
    }

    #[test]
    fn render_ranks_multiple_by_speedup_desc() {
        let make = |sp: f64, id: &str| {
            let mut o = canonical_opp();
            o.estimated_speedup = Some(sp);
            o.location = Some(FusionLocation {
                fx_node_ids: vec![id.to_string()],
                src_lineno_range: None,
            });
            o
        };
        // Intentionally unsorted input; renderer must rank highest speedup first (also AC #3: ≥3).
        let md = render(
            &[make(1.20, "c"), make(2.04, "a"), make(1.50, "b")],
            Format::Markdown,
        );
        assert!(md.contains("### #1:") && md.contains("### #2:") && md.contains("### #3:"));
        let (p_high, p_mid, p_low) = (
            md.find("2.04").unwrap(),
            md.find("1.50").unwrap(),
            md.find("1.20").unwrap(),
        );
        assert!(
            p_high < p_mid && p_mid < p_low,
            "ranked by speedup descending"
        );
    }

    #[test]
    fn render_is_honest_when_cost_absent() {
        let mut opp = canonical_opp();
        opp.shape = None;
        opp.baseline_hbm_bytes = None;
        opp.fused_hbm_bytes = None;
        opp.estimated_speedup = None;
        let md = render(&[opp], Format::Markdown);
        assert!(md.contains("not estimated"), "no fabricated cost");
        assert!(md.contains("GEMM-Residual-RMSNorm-GEMM"), "still reported");
        assert!(md.contains("Confidence: high"));
    }

    #[test]
    fn render_confidence_medium_for_decomposed() {
        let mut opp = canonical_opp();
        opp.confidence = Some("medium".to_string());
        let md = render(&[opp], Format::Markdown);
        assert!(md.contains("Confidence: medium"));
    }

    #[test]
    fn render_small_graph_shows_sub_mb_not_zero() {
        // A toy graph's traffic is well under 1 MB; it must read as decimals (0.21 MB), never a
        // misleading "0 MB" (surfaced by the Tool 6 end-to-end validation on a tiny capture).
        let mut opp = canonical_opp();
        opp.baseline_hbm_bytes = Some(212_992.0);
        opp.fused_hbm_bytes = Some(131_584.0);
        opp.estimated_speedup = Some(1.62);
        let md = render(&[opp], Format::Markdown);
        assert!(md.contains("Baseline HBM traffic: 0.21 MB"), "md:\n{md}");
        assert!(md.contains("CODA-fused HBM traffic: 0.13 MB"), "md:\n{md}");
        assert!(
            !md.contains(" 0 MB"),
            "sub-MB must not round to a misleading 0 MB:\n{md}"
        );
    }

    fn with_speedup(s: Option<f64>) -> FusionOpportunity {
        let mut o = canonical_opp();
        o.estimated_speedup = s;
        o
    }

    #[test]
    fn select_drops_below_min_but_keeps_unestimated() {
        // 1.1 < min(1.5) → dropped; 2.0 → kept; no estimate → kept (real finding, just unrankable).
        let kept = select(
            vec![
                with_speedup(Some(1.1)),
                with_speedup(Some(2.0)),
                with_speedup(None),
            ],
            1.5,
            10,
        );
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].estimated_speedup, Some(2.0), "ranked first");
        assert!(
            kept.iter().any(|o| o.estimated_speedup.is_none()),
            "unestimated kept"
        );
    }

    #[test]
    fn select_ranks_then_truncates_to_top_k() {
        let kept = select(
            vec![
                with_speedup(Some(2.0)),
                with_speedup(Some(3.0)),
                with_speedup(Some(1.5)),
            ],
            1.0,
            2,
        );
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].estimated_speedup, Some(3.0));
        assert_eq!(kept[1].estimated_speedup, Some(2.0));
    }
}
