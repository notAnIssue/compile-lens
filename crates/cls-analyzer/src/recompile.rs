//! Tool 1 — recompile aggregator. Skeleton only.
//!
//! The analyzer consumes a parsed [`Session`] artifact and produces a
//! [`RecompileFindings`] summary: total recompilation count, the guard
//! expression clusters the events fall into, and a small ranked list of
//! suggestions a user can act on (e.g. *"mark dimension 0 dynamic"*).
//!
//! This module ships the **type surface** Phase 1's clustering and
//! suggestion PRs will fill in. [`analyze`] returns
//! [`ClsError::NotYetImplemented`] today; once the algorithm work lands,
//! callers see real findings without needing to change the call site.
//!
//! See ADR-029 for why source attribution is derived in the analyzer
//! rather than emitted by the collector at the v0.5.0 schema.

use cls_errors::ClsError;
use cls_schema::Session;

/// The summary the analyzer produces from a session's recompile events.
///
/// Cluster + suggestion shapes are intentionally small — the algorithm PRs
/// can grow fields as needed without breaking the call signature, because
/// the struct is not part of the on-disk schema (that contract lives in
/// `cls-schema`; this struct is the Rust-internal result type).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecompileFindings {
    /// Number of `Recompilation` events the analyzer observed in the
    /// session. Zero on an empty session.
    pub total_recompilations: u64,

    /// Guard expression clusters the recompiles group into. Empty until
    /// the clustering PR lands.
    pub guard_categories: Vec<GuardCategory>,

    /// Top-ranked actionable suggestions for the user. Empty until the
    /// suggestion-generation PR lands.
    pub top_suggestions: Vec<Suggestion>,
}

/// A single guard expression cluster.
///
/// Fields are deliberately minimal in the skeleton; clustering adds
/// `values_observed`, `source_locations`, and similar as the algorithm
/// is implemented.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardCategory {
    /// Coarse category label — `"shape"`, `"dtype"`, `"stride"`,
    /// `"callable"`, `"unknown"`. Open string for forward compat with
    /// PyTorch's guard taxonomy.
    pub category: String,

    /// Number of recompile events in this cluster.
    pub count: u64,
}

/// A single ranked suggestion.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// Short, action-oriented text.
    pub text: String,
}

/// Run Tool 1 on a parsed session.
///
/// Returns [`ClsError::NotYetImplemented`] until the clustering PR lands.
/// Empty sessions take an early return so test harnesses can wire a
/// happy-path call without conditional logic, but every non-empty session
/// is treated as not-yet-implemented work.
#[tracing::instrument(skip(session), fields(recompile_count = session.iteration_count.unwrap_or(0)))]
pub fn analyze(session: &Session) -> Result<RecompileFindings, ClsError> {
    // The session itself does not own `recompilations[]` — those live on
    // the parent `ClsArtifact`. Accepting `&Session` here keeps the
    // signature stable across the schema's nesting model; the upcoming
    // PRs will switch to `&ClsArtifact` if richer access is required.
    let recompile_count_hint = session.iteration_count.unwrap_or(0);

    if recompile_count_hint == 0 {
        // Truly empty session — let callers exercise the surface without
        // hitting the not-yet-implemented branch. Once the algorithm
        // lands, this short-circuit goes away.
        return Ok(RecompileFindings::default());
    }

    Err(ClsError::NotYetImplemented {
        surface: "RecompileAnalyzer::analyze".into(),
        tracking: "Phase 1 (Tool 1) clustering + suggestion PRs".into(),
    })
}
