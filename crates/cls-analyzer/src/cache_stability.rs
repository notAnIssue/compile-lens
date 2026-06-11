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

/// The diff-based (Mode A) cache-stability result between a base and head session. It catches a
/// *regression a change introduced*: the head run becomes unstable where the base run was steady,
/// or the head triggers recompilations the base did not.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CacheStabilityDiff {
    /// The head iteration at which the output first changed while the base stayed stable across all
    /// its iterations. `None` if there is no such regression — either the base was already unstable
    /// (so any head instability is not attributable to the change), or the head stayed stable too.
    pub output_instability_introduced: Option<u64>,
    /// Iterations at which the head triggered a recompilation the base did not (by `iteration_index`).
    pub new_recompilations: Vec<u64>,
}

impl CacheStabilityDiff {
    /// True when the diff found no cache-stability regression.
    pub fn is_clean(&self) -> bool {
        self.output_instability_introduced.is_none() && self.new_recompilations.is_empty()
    }
}

/// Mode A: compare a base and head session's `iterations[]` for cache-stability regressions.
pub fn analyze_diff(before: &ClsArtifact, after: &ClsArtifact) -> CacheStabilityDiff {
    analyze_diff_iterations(&before.iterations, &after.iterations)
}

/// The first iteration whose output signature differs from the previous one (both known), or `None`
/// if the run's output stayed stable across every iteration. As in Mode B, only a *known* inequality
/// counts — an absent signature is unknown, not a change.
fn first_output_change(iterations: &[Iteration]) -> Option<u64> {
    for i in 1..iterations.len() {
        if let (Some(a), Some(b)) = (
            &iterations[i].output_signature,
            &iterations[i - 1].output_signature,
        ) {
            if a != b {
                return Some(iterations[i].iteration_index);
            }
        }
    }
    None
}

/// The Mode A algorithm, decoupled from the artifact shape for inline testing.
pub(crate) fn analyze_diff_iterations(
    before: &[Iteration],
    after: &[Iteration],
) -> CacheStabilityDiff {
    // Output-stability regression: attribute head instability to the change only if the base was
    // steady across all its iterations. If the base already changed its output, head instability is
    // not a regression the change introduced.
    let output_instability_introduced = match first_output_change(before) {
        Some(_) => None,
        None => first_output_change(after),
    };

    // Recompilations the head triggered that the base did not, by iteration index.
    let base_recompiles: std::collections::BTreeSet<u64> = before
        .iter()
        .filter(|it| it.recompilation_triggered == Some(true))
        .map(|it| it.iteration_index)
        .collect();
    let new_recompilations: Vec<u64> = after
        .iter()
        .filter(|it| {
            it.recompilation_triggered == Some(true)
                && !base_recompiles.contains(&it.iteration_index)
        })
        .map(|it| it.iteration_index)
        .collect();

    CacheStabilityDiff {
        output_instability_introduced,
        new_recompilations,
    }
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

    // --- Mode A (diff-based) ---

    /// Mark an iteration as having triggered a recompilation.
    fn iter_recompiled(idx: u64, output_sig: Option<&str>) -> Iteration {
        let mut it = iter(idx, None, output_sig, &[]);
        it.recompilation_triggered = Some(true);
        it
    }

    /// Base output stable across all iterations, head output changes at iteration 2 → regression
    /// introduced at 2.
    #[test]
    fn test_output_instability_introduced() {
        let base = [iter(0, None, Some("A"), &[]), iter(1, None, Some("A"), &[])];
        let head = [
            iter(0, None, Some("A"), &[]),
            iter(1, None, Some("A"), &[]),
            iter(2, None, Some("B"), &[]),
        ];
        let d = analyze_diff_iterations(&base, &head);
        assert_eq!(d.output_instability_introduced, Some(2));
        assert!(!d.is_clean());
    }

    /// Base was already unstable → head instability is not a regression the change introduced.
    #[test]
    fn test_base_already_unstable_is_not_a_regression() {
        let base = [iter(0, None, Some("A"), &[]), iter(1, None, Some("B"), &[])];
        let head = [iter(0, None, Some("A"), &[]), iter(1, None, Some("C"), &[])];
        assert_eq!(
            analyze_diff_iterations(&base, &head).output_instability_introduced,
            None
        );
    }

    /// Head recompiles at iteration 2 where the base did not → flagged.
    #[test]
    fn test_new_recompilation_detected() {
        let base = [
            iter(0, None, Some("A"), &[]),
            iter(1, None, Some("A"), &[]),
            iter(2, None, Some("A"), &[]),
        ];
        let head = [
            iter(0, None, Some("A"), &[]),
            iter(1, None, Some("A"), &[]),
            iter_recompiled(2, Some("A")),
        ];
        assert_eq!(
            analyze_diff_iterations(&base, &head).new_recompilations,
            vec![2]
        );
    }

    /// A recompilation both sides share is not a regression.
    #[test]
    fn test_shared_recompilation_not_flagged() {
        let base = [iter(0, None, Some("A"), &[]), iter_recompiled(1, Some("A"))];
        let head = [iter(0, None, Some("A"), &[]), iter_recompiled(1, Some("A"))];
        assert!(analyze_diff_iterations(&base, &head)
            .new_recompilations
            .is_empty());
    }

    /// Both stable, same recompiles → clean diff.
    #[test]
    fn test_clean_diff() {
        let base = [iter(0, None, Some("A"), &[]), iter(1, None, Some("A"), &[])];
        let head = [iter(0, None, Some("A"), &[]), iter(1, None, Some("A"), &[])];
        assert!(analyze_diff_iterations(&base, &head).is_clean());
    }
}
