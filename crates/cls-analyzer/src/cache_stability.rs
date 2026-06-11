//! Tool 2b — cache-stability detection (Mode B: single-run anomaly).
//!
//! A `torch.compile` cache-stability bug is one of the hardest to catch by hand: the model's
//! internal state changes between iterations (a buffer or scalar attribute drifts), the compiled
//! graph is reused from cache anyway, and the output is *frozen* — identical to the previous
//! iteration when it should have moved. The numerics are silently wrong while the run looks fine.
//! This is the cache-stability analogue of Tool 2a's operand-order false-negative (Li et al. 2026
//! §3.2.1, Listing 2).
//!
//! Mode B works on a single run's `iterations[]` (already captured by the Tool 2a collector — no
//! new capture). For each iteration after the first, three conditions together are the signature:
//!
//! - **state_mutated** — `internal_state_snapshot.module_attrs_changed` is non-empty (something in
//!   the module's mutable state drifted this iteration),
//! - **cache_reused** — `cache_hit` is true (the compiled graph was served from cache), and
//! - **output_frozen** — this iteration's `output_signature` equals the previous one's (the output
//!   did not move).
//!
//! All three at once is the bug: state changed, cache didn't invalidate, output stuck. (Mode A —
//! the diff-based base-vs-head check — lands in a later change.)

use cls_schema::{ClsArtifact, Iteration};

/// Severity of a cache-stability finding. Only `High` exists today — the three-condition signature
/// is specific enough that a match is a real, high-severity suspicion, not a soft warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    High,
}

/// One cache-stability finding: a single iteration that matched the mutable-state-not-invalidated
/// signature.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CacheStabilityFinding {
    pub severity: Severity,
    /// Stable machine-readable pattern id.
    pub pattern: String,
    /// The iteration (`iteration_index`) at which the signature matched.
    pub iteration_index: u64,
    /// The module attributes that drifted while the cache stayed reused and the output frozen.
    pub changed_attrs: Vec<String>,
    /// Citation for the pattern.
    pub reference: String,
}

/// The full set of cache-stability findings for one session.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CacheStabilityFindings {
    pub findings: Vec<CacheStabilityFinding>,
}

/// Analyze a parsed artifact's `iterations[]` for the Mode B cache-stability signature. Pure: the
/// same iterations always produce the same findings, and there is no error path (it only reads
/// structured fields).
pub fn analyze(artifact: &ClsArtifact) -> CacheStabilityFindings {
    analyze_iterations(&artifact.iterations)
}

/// The algorithm, decoupled from the artifact shape so it can be tested on inline `Iteration`
/// vectors without constructing a whole `ClsArtifact`.
pub(crate) fn analyze_iterations(iterations: &[Iteration]) -> CacheStabilityFindings {
    let mut findings = Vec::new();

    for i in 1..iterations.len() {
        let cur = &iterations[i];
        let prev = &iterations[i - 1];

        // state_mutated: some module attribute drifted this iteration.
        let changed_attrs = cur
            .internal_state_snapshot
            .as_ref()
            .map(|s| s.module_attrs_changed.clone())
            .unwrap_or_default();
        let state_mutated = !changed_attrs.is_empty();

        // cache_reused: the compiled graph was served from cache (not a fresh compile).
        let cache_reused = cur.cache_hit == Some(true);

        // output_frozen: output identical to the previous iteration. Only a *known* equality counts
        // — if either signature is absent we cannot claim the output froze.
        let output_frozen = match (&cur.output_signature, &prev.output_signature) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        };

        if state_mutated && cache_reused && output_frozen {
            findings.push(CacheStabilityFinding {
                severity: Severity::High,
                pattern: "graph_caching_mutable_state_not_invalidated".to_string(),
                iteration_index: cur.iteration_index,
                changed_attrs,
                reference: "Li et al. 2026 §3.2.1 + Listing 2".to_string(),
            });
        }
    }

    CacheStabilityFindings { findings }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cls_schema::InternalStateSnapshot;

    /// Build an `Iteration` with just the fields Mode B reads.
    fn iter(
        idx: u64,
        cache_hit: Option<bool>,
        output_sig: Option<&str>,
        changed: &[&str],
    ) -> Iteration {
        Iteration {
            iteration_index: idx,
            timestamp_ms: None,
            guard_evaluations: vec![],
            cache_hit,
            recompilation_triggered: None,
            output_signature: output_sig.map(String::from),
            internal_state_snapshot: Some(InternalStateSnapshot {
                module_buffers_hash: None,
                module_attrs_changed: changed.iter().map(|s| s.to_string()).collect(),
            }),
        }
    }

    /// Listing 2: state drifts, cache is reused, output frozen → one High finding.
    #[test]
    fn test_listing2_pattern_is_high() {
        let iters = [
            iter(0, Some(false), Some("sigA"), &[]),
            iter(1, Some(true), Some("sigA"), &["step_counter"]),
        ];
        let out = analyze_iterations(&iters);
        assert_eq!(out.findings.len(), 1);
        let f = &out.findings[0];
        assert_eq!(f.severity, Severity::High);
        assert_eq!(f.iteration_index, 1);
        assert_eq!(f.changed_attrs, vec!["step_counter"]);
        assert_eq!(f.pattern, "graph_caching_mutable_state_not_invalidated");
    }

    /// Normal stateful module: state changed but the cache correctly *missed* (recompiled), so no
    /// finding — the cache did its job.
    #[test]
    fn test_state_change_with_cache_miss_is_silent() {
        let iters = [
            iter(0, Some(false), Some("sigA"), &[]),
            iter(1, Some(false), Some("sigB"), &["step_counter"]),
        ];
        assert!(analyze_iterations(&iters).findings.is_empty());
    }

    /// State changed and cache reused, but the output *did* move → not frozen, no bug.
    #[test]
    fn test_output_moved_is_silent() {
        let iters = [
            iter(0, Some(true), Some("sigA"), &[]),
            iter(1, Some(true), Some("sigB"), &["step_counter"]),
        ];
        assert!(analyze_iterations(&iters).findings.is_empty());
    }

    /// Cache reused and output frozen, but no state drift → expected steady state, no bug.
    #[test]
    fn test_no_state_drift_is_silent() {
        let iters = [
            iter(0, Some(true), Some("sigA"), &[]),
            iter(1, Some(true), Some("sigA"), &[]),
        ];
        assert!(analyze_iterations(&iters).findings.is_empty());
    }

    /// Missing signatures cannot establish output_frozen → no false positive.
    #[test]
    fn test_missing_output_signature_is_silent() {
        let iters = [
            iter(0, Some(true), None, &[]),
            iter(1, Some(true), None, &["step_counter"]),
        ];
        assert!(analyze_iterations(&iters).findings.is_empty());
    }

    /// Fewer than two iterations: nothing to compare.
    #[test]
    fn test_single_iteration_is_silent() {
        let iters = [iter(0, Some(true), Some("sigA"), &["step_counter"])];
        assert!(analyze_iterations(&iters).findings.is_empty());
        assert!(analyze_iterations(&[]).findings.is_empty());
    }

    /// Several offending iterations → several findings, each pinned to its iteration_index.
    #[test]
    fn test_multiple_findings() {
        let iters = [
            iter(0, Some(false), Some("sigA"), &[]),
            iter(1, Some(true), Some("sigA"), &["a"]),
            iter(2, Some(true), Some("sigA"), &["b"]),
        ];
        let out = analyze_iterations(&iters);
        assert_eq!(out.findings.len(), 2);
        assert_eq!(out.findings[0].iteration_index, 1);
        assert_eq!(out.findings[1].iteration_index, 2);
    }
}
