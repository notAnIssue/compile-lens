//! Tool 1 — recompile aggregator.
//!
//! [`analyze`] consumes a parsed [`ClsArtifact`] and produces a [`RecompileFindings`] summary:
//! the total recompilation count and the guard clusters the events fall into. Following the
//! "describe → attribute" north star (ADR-032), a cluster is not a coarse bucket — it attributes
//! the recompiles to a concrete **dynamic axis** (e.g. `x[0]`) with the value transitions that
//! drove them, so the suggestion PR can emit an axis-precise fix (`mark_dynamic(x, 0)`).
//!
//! The clustering algorithm lives in [`crate::recompile_cluster`]; ranked suggestions land in a
//! later PR (`top_suggestions` is empty until then). Source attribution is derived here rather
//! than emitted by the collector at the v0.5.0 schema (ADR-029).
//!
//! Why no algorithm ADR: the generic "token vs AST vs edit-distance clustering" choice the plan
//! anticipated dissolved once ADR-032 reframed the input as structured mismatch text — the guard
//! already names its category, axis, and value transition, so we parse it directly. The only
//! fuzziness left (the `other` bucket) is handled by literal canonicalization. One obvious
//! approach won, so the choice is recorded here rather than in a standalone ADR.

use cls_errors::ClsError;
use cls_schema::ClsArtifact;

use crate::recompile_cluster;

/// The summary the analyzer produces from a session's recompile events.
///
/// This is the Rust-internal result type (not part of the on-disk schema, which lives in
/// `cls-schema`), so fields can grow without an ADR-027 schema rev.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecompileFindings {
    /// Number of `Recompilation` events observed. Zero on an empty artifact.
    pub total_recompilations: u64,

    /// Guard clusters the recompiles group into, ordered by descending frequency.
    pub guard_categories: Vec<GuardCategory>,

    /// Top-ranked actionable suggestions. Empty until the suggestion-generation PR lands.
    pub top_suggestions: Vec<Suggestion>,
}

/// A cluster of recompiles attributed to one guard / dynamic axis.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardCategory {
    /// Coarse category — `"size"`, `"dtype"`, `"stride"`, or `"other"`. Open string for forward
    /// compat with PyTorch's guard taxonomy.
    pub category: String,

    /// Number of recompile events in this cluster.
    pub count: u64,

    /// Canonical guard template the cluster's members share, e.g.
    /// `"tensor '<t>' size mismatch at index <n>"`. The grouping key alongside `axis`.
    pub canonical_guard: String,

    /// The dynamic axis the recompiles attribute to, e.g. `"x[0]"` (tensor + dim) or `"x"`
    /// (whole-tensor, for dtype). `None` for an `other` guard with no extractable axis.
    pub axis: Option<String>,

    /// Distinct `previous -> new` value transitions observed for this cluster (e.g.
    /// `4 -> 8`, `8 -> 16`). Empty when the guard carried no values.
    pub observed_values: Vec<ValueTransition>,
}

/// One observed guard value change (the `expected` / `actual` of a failed guard).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueTransition {
    pub previous: Option<String>,
    pub new: Option<String>,
}

/// A single ranked suggestion.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// Short, action-oriented text.
    pub text: String,
}

/// Run Tool 1 on a parsed artifact.
///
/// The recompile events live on [`ClsArtifact::recompilations`] (the parent artifact, not the
/// nested `Session`). Returns default (empty) findings for an artifact with no recompiles. Does
/// not error on malformed guards — a recompile with no guard expression is counted in
/// `total_recompilations` but contributes no cluster.
#[tracing::instrument(skip(artifact), fields(recompile_count = artifact.recompilations.len()))]
pub fn analyze(artifact: &ClsArtifact) -> Result<RecompileFindings, ClsError> {
    Ok(RecompileFindings {
        total_recompilations: artifact.recompilations.len() as u64,
        guard_categories: recompile_cluster::cluster(&artifact.recompilations),
        top_suggestions: Vec::new(),
    })
}
