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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_clock_ms: Option<f64>,
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
}

/// One named compile phase with its wall-clock cost (`compile_phases[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilePhase {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
