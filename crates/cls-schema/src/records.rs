//! The normalized top-level record arrays (everything except `kernels` /
//! `roofline_predictions`, which live in [`crate::kernels`]) plus their shared location
//! types.
//!
//! These structs intentionally do **not** carry an `extra` capture map: unknown keys are
//! accepted (no deserialize error) but not preserved on re-serialize. The forward-compat
//! capture is reserved for the two growth points in [`crate`] (`ClsArtifact`, `Session`);
//! see the crate-level docs.

use serde::{Deserialize, Serialize};

/// A single source position (used by [`GraphBreak`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLocation {
    /// Scrubbed or relative path per `redaction_policy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

/// A source span (start..=end), used by [`LintFinding`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRange {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_start: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_end: Option<u64>,
    /// Opt-in only; omitted by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_excerpt: Option<String>,
}

/// A point where Dynamo fell back to eager (`graph_breaks[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphBreak {
    pub break_id: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op_or_construct: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_kind: Option<String>,
}

/// The guard whose failure triggered a recompilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailedGuard {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_value: Option<String>,
}

/// A recompilation event (`recompilations[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recompilation {
    pub recompilation_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled_function_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_guard: Option<FailedGuard>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurred_at_step: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub wall_clock_ms: Option<f64>,
    /// Source file:line of the recompiled function (the compiled region), parsed from torch's
    /// recompile log. Lets the summary point at where the recompiling region is defined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_location: Option<SourceLocation>,
}

/// A compiled graph artifact (`compiled_graphs[]`). Cross-references `kernels[]` and guards
/// by id (normalized layout, ADR-021).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledGraph {
    pub graph_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled_function_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fx_graph_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inductor_ir_path: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guard_list: Vec<String>,
    /// Cross-reference into `kernels[]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kernel_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "indexmap::IndexMap::is_empty")]
    pub compile_phases_summary: crate::JsonMap,
    /// Node-level FX graph structure consumed by the WL-signature diff (Tool 2a, ADR-024).
    /// Inlined into the artifact so a single `.cls.json` stays self-contained rather than
    /// pointing at a side file. Optional and forward-compatible: a Tool 1 artifact written
    /// before this field existed simply omits it (empty), and the diff treats a graph with
    /// no `nodes` as "structure not captured".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<FxNode>,
}

/// One node of a captured FX graph (`compiled_graphs[].nodes[]`).
///
/// An FX graph is the computation graph `torch.compile` traces from model code: a sequence
/// of operations (nodes) with dependency edges. Edges are encoded implicitly — a node lists
/// the ids of the upstream nodes whose outputs it consumes in [`FxNode::inputs`]. This is the
/// minimum structure the WL-signature diff needs to tell which nodes a change added, removed,
/// or modified (ADR-024).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FxNode {
    /// Graph-unique node id. Other nodes reference it in their `inputs` to encode an edge.
    pub id: String,
    /// The operation, e.g. `aten.matmul` or a `call_function` target.
    pub op_type: String,
    /// Ordered ids of the upstream nodes this one consumes. **Order is load-bearing**:
    /// `sub(a, b)` and `sub(b, a)` differ only in this ordering, so operand-order regressions
    /// are detectable only because the sequence is preserved rather than treated as a set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<String>,
    /// Node attributes (a constant's value, a dim parameter, …). Two otherwise-identical nodes
    /// that differ here are classified `modified` by the diff.
    #[serde(default, skip_serializing_if = "indexmap::IndexMap::is_empty")]
    pub attrs: crate::JsonMap,
    /// The node's output tensor shape, when the collector captured a *concrete* one from
    /// `node.meta['val']`. Empty when unknown — meta absent, the value isn't a single tensor, or
    /// the shape is dynamic (symbolic `SymInt`, which has no byte size). Tool 6's cost model reads
    /// these to size HBM traffic; structure-only consumers (the WL-diff) ignore them. Added after
    /// the collector's original contract (ADR-024) — older artifacts simply omit it (ADR-038).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub out_shape: Vec<u64>,
    /// The node's output dtype as a torch name with the `torch.` prefix stripped (`bfloat16`,
    /// `float32`). Absent under the same conditions as `out_shape`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_dtype: Option<String>,
    /// Source file of the node's defining line — the innermost user frame of
    /// `node.meta['stack_trace']`, path-normalized at collection (D11). Absent when the trace was
    /// unavailable. Lets the diff and fusion sections point a finding at the exact source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    /// 1-based line of the node's defining source, paired with `source_file`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_line: Option<u64>,
}

/// One named compile phase with its wall-clock cost (`compile_phases[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilePhase {
    pub name: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub duration_ms: Option<f64>,
}

/// One guard evaluation observed during an iteration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardEvaluation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_result: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_value: Option<String>,
}

/// Snapshot of mutable module state, used to catch cache-stability bugs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InternalStateSnapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_buffers_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub module_attrs_changed: Vec<String>,
}

/// One execution iteration (`iterations[]`), supporting multi-iteration capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Iteration {
    pub iteration_index: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub timestamp_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guard_evaluations: Vec<GuardEvaluation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recompilation_triggered: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub internal_state_snapshot: Option<InternalStateSnapshot>,
}

/// The upstream PyTorch issue a lint finding maps to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceIssue {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_in_version: Option<String>,
}

/// A compile-lint hit (`lint_findings[]`).
///
/// `pattern_category` and `severity` stay `String` rather than enums: the v0 pattern
/// vocabulary is documented to grow post-v1.0, and a closed enum would reject a
/// generously-collected future category (D6). See the design note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LintFinding {
    pub finding_id: String,
    pub pattern_category: String,
    pub severity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_location: Option<SourceRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_pattern: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub li_et_al_section: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_issue: Option<ReferenceIssue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workaround: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applies_to_user_torch_version: Option<bool>,
}

/// One eager-vs-compiled numerical divergence finding (`divergences[]`), produced by Tool 3.
///
/// A normalized record (ADR-021, ADR-034): a divergence is an *analysis result*, so — like
/// `lint_findings`, and unlike the single-object `compile_config` run-metadata — it lives in a
/// top-level array and carries a `divergence_id`. The causal attribution is **nested** rather than
/// a separate id-joined array, because it is strictly subordinate to this one finding (ADR-034).
///
/// Fields mirror the Python `DivergenceFindings` 1:1. `first_divergent_layer` is absent when nothing
/// diverged; `max_abs_diff` is absent for a shape mismatch (where it is undefined) or no divergence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Divergence {
    pub divergence_id: String,
    /// Qualified name of the first layer (in eager execution order) that disagreed beyond
    /// tolerance; absent when nothing diverged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_divergent_layer: Option<String>,
    /// Max absolute element-wise difference at that layer; absent for a shape mismatch or when
    /// nothing diverged.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub max_abs_diff: Option<f64>,
    /// How many layers were numerically compared (both sides present and tensor-valued).
    pub num_layers_compared: u64,
    pub rtol: f64,
    pub atol: f64,
    /// Human-readable attributed cause; absent when the causal experiment did not run. When an
    /// `attribution` is present this is its `summary` (a denormalized convenience, ADR-034).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_cause: Option<String>,
    /// Structured pass-level causal attribution; absent when the causal experiment did not run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<DivergenceAttribution>,
}

/// The pass-level causal attribution for a [`Divergence`] (nested; ADR-032 / ADR-034).
///
/// Pass-level only: `torch._inductor.config` exposes pass-level toggles, not per-node fusion
/// control, so `responsible_passes` names the inductor pass(es) whose disabling removes the
/// divergence — never a specific fused node. Mirrors the Python `CausalAttribution` 1:1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DivergenceAttribution {
    /// True iff disabling `responsible_passes` made eager and compiled agree.
    pub attributed: bool,
    /// Minimal set of inductor passes whose disabling removes the divergence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub responsible_passes: Vec<String>,
    pub summary: String,
    /// How many recompile-and-recheck probes the experiment ran (cost transparency).
    pub num_probes: u64,
}

/// Tool 6 algebraic fusion opportunity (`fusion_opportunities[]`). An analysis result: a matched
/// `GEMM-Residual-RMSNorm-GEMM`-style pattern with its analytical HBM-traffic estimate and a
/// plain-language description of the fusion to apply. Suggest-only — the detector identifies and
/// quantifies the opportunity; it never writes, imports, or runs a kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionOpportunity {
    /// Matched pattern, e.g. `"A"` (GEMM-Residual-RMSNorm-GEMM).
    pub pattern_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<FusionLocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<FusionShape>,
    /// Standard-path HBM traffic in bytes (analytical roofline estimate, not measured).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub baseline_hbm_bytes: Option<f64>,
    /// CODA-fused HBM traffic in bytes (analytical roofline estimate).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub fused_hbm_bytes: Option<f64>,
    /// `baseline / fused` under the memory-bound assumption; analytical only (N10).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::non_finite::opt"
    )]
    pub estimated_speedup: Option<f64>,
    /// Plain-language description of the fusion to apply, e.g. "fold per-row 1/rms into the second
    /// GEMM epilogue". Suggest-only — names no kernel, library, or API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_fusion: Option<String>,
    /// `high` / `medium` / `low`. Kept `String` for forward-compat (same as the lint vocabularies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<String>,
}

/// Where a fusion opportunity lives, in the graph and (when localizable) the source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionLocation {
    /// The FX node ids forming the matched pattern.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fx_node_ids: Vec<String>,
    /// Source location (`file:line-line`) via Tool 1 localization, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_lineno_range: Option<String>,
}

/// The GEMM shapes a fusion opportunity spans; the cost model reads these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionShape {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub m: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k0: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k1: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
}
