//! D7 release-blocker invariant for Tool 5: the rank-correlation pruning gate is **sound**.
//!
//! Autotune pruning is only safe if it prunes *exactly when* the predicted ranking actually tracks
//! the measured one. The release-blocker property guarded here is that soundness:
//!
//!   * a well-ranked grid (high Spearman) is trusted and pruned to the top-K;
//!   * an anti-correlated grid (the predictor is backwards) is **never** pruned — full sweep;
//!   * a degenerate grid where the model predicts every config identically (it has no ranking
//!     signal — e.g. a memory-bound kernel) yields *no* decision, so the caller falls back to a
//!     full sweep rather than pruning on noise;
//!   * the trust threshold for aggressive pruning stays at ≥ 0.8 (and ≥ 0.5 for partial), so a
//!     future change cannot silently lower the bar at which configs start getting dropped.
//!
//! A red check here means "do not release": the cost model would prune configs on an untrustworthy
//! ranking, which can drop the genuinely fastest one. These are *deterministic* properties over
//! controlled predicted/measured pairs, mirroring the compile-diff D7 invariants — they need no GPU.
//!
//! Note on scope: whether the model ranks any *particular* real kernel well enough to prune is
//! empirical and kernel-dependent (a memory-bound kernel gives it no signal; a GPU smoke on real
//! hardware shows the gate correctly falling back to a full sweep there). The invariant
//! is the gate's *soundness*, not a universal "ranks every kernel ≥ 0.7" claim — which would be
//! false, and which the safe-fallback tier exists precisely to handle.

use cls_roofline::{prune_mode, pruning_decision, spearman_correlation, PruneMode};

#[test]
fn well_ranked_grid_is_trusted_and_pruned() {
    // Predicted order == measured order (monotone) → Spearman 1.0 → aggressive → measure only top-K.
    let predicted = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let measured = [1.1, 2.2, 2.9, 4.3, 5.0, 6.1, 7.2, 8.0]; // monotone with predicted
    let rho = spearman_correlation(&predicted, &measured).expect("non-degenerate series");
    assert!(
        rho >= 0.7,
        "a monotone grid must clear the 0.7 trust bar, got {rho}"
    );

    let decision = pruning_decision(&predicted, &measured, 2).expect("a decision");
    assert_eq!(decision.mode.as_deref(), Some("aggressive"));
    assert_eq!(decision.configs_total, Some(8));
    assert_eq!(
        decision.configs_measured,
        Some(2),
        "8 configs pruned to the top 2"
    );
}

#[test]
fn anticorrelated_grid_is_never_pruned() {
    // The predictor is backwards: predicted ascending, measured descending → Spearman -1.0. Trusting
    // it would measure the *slowest* configs and miss the fastest, so the gate must measure all.
    let predicted = [10.0, 20.0, 30.0, 40.0];
    let measured = [4.0, 3.0, 2.0, 1.0];
    let rho = spearman_correlation(&predicted, &measured).expect("non-degenerate series");
    assert!(
        rho < 0.5,
        "an anti-correlated grid must not be trusted, got {rho}"
    );

    let decision = pruning_decision(&predicted, &measured, 1).expect("a decision");
    assert_eq!(
        decision.mode.as_deref(),
        Some("disabled_fallback_full_sweep")
    );
    assert_eq!(
        decision.configs_measured,
        Some(4),
        "nothing pruned — every config measured"
    );
}

#[test]
fn degenerate_ranking_yields_no_decision_so_the_caller_full_sweeps() {
    // A memory-bound kernel the model can't rank: every config predicted identically (observed on
    // real hardware). Spearman is undefined on a constant series, so there is no pruning decision —
    // the caller must fall back to a full sweep rather than prune on noise.
    let predicted = [20.0, 20.0, 20.0, 20.0];
    let measured = [70.0, 74.0, 65.0, 90.0];
    assert!(
        spearman_correlation(&predicted, &measured).is_none(),
        "a constant predicted series has no rank correlation"
    );
    assert!(
        pruning_decision(&predicted, &measured, 1).is_none(),
        "no correlation → no pruning decision → caller full-sweeps"
    );
}

#[test]
fn the_aggressive_trust_bar_cannot_silently_drop() {
    // Guard the tier thresholds: aggressive needs ≥ 0.8, partial (moderate) needs ≥ 0.5, below that
    // the gate disables. A change that lowered these would start pruning on weaker evidence.
    assert_eq!(prune_mode(0.80), PruneMode::Aggressive);
    assert_eq!(prune_mode(0.79), PruneMode::Moderate);
    assert_eq!(prune_mode(0.50), PruneMode::Moderate);
    assert_eq!(prune_mode(0.49), PruneMode::DisabledFallbackFullSweep);
}
