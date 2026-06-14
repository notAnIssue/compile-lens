//! Tool 6 — CODA-style algebraic fusion-opportunity detector (crown-jewel feature).
//!
//! Finds `GEMM-Residual-RMSNorm-GEMM` (Pattern A) fusion opportunities in a `torch.compile` FX
//! graph that Inductor leaves on the table — cross-module algebraic fusions its local schedule
//! rules don't perform (the `1/rms` row-constant scale can be pulled through the second GEMM and
//! applied in its epilogue, removing the full-tensor RMSNorm materialization between the two GEMMs).
//! It is **suggest-only**: it reads the serialized graph, never mutates it, and never runs a kernel.
//!
//! Three disciplines bound the tool (design source-of-truth: `the design notes`):
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

use cls_schema::{ClsArtifact, FusionLocation, FusionOpportunity, FxNode};

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
/// node range, the suggested kernel, and the confidence; the shape and HBM-traffic estimate are
/// filled by the cost model in a later change.
pub fn analyze(artifact: &ClsArtifact) -> Vec<FusionOpportunity> {
    let mut out = Vec::new();
    for graph in &artifact.compiled_graphs {
        for m in find_pattern_a(&graph.nodes) {
            let mut fx_node_ids = vec![m.gemm1, m.add];
            fx_node_ids.extend(m.rms_norm_ids);
            fx_node_ids.push(m.gemm2);
            out.push(FusionOpportunity {
                pattern_id: "A".to_string(),
                location: Some(FusionLocation {
                    fx_node_ids,
                    src_lineno_range: None,
                }),
                shape: None,
                baseline_hbm_bytes: None,
                fused_hbm_bytes: None,
                estimated_speedup: None,
                suggested_kernel: Some("fold per-row 1/rms into the second GEMM epilogue".to_string()),
                // Native exact match is high-confidence; a decomposed subgraph match is medium.
                confidence: Some(if m.decomposed { "medium" } else { "high" }.to_string()),
            });
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
            found[0].suggested_kernel.as_deref(),
            Some("fold per-row 1/rms into the second GEMM epilogue")
        );
        let ids = &found[0].location.as_ref().unwrap().fx_node_ids;
        assert_eq!(ids, &vec!["g1", "add", "rms", "g2"]);
    }
}
