//! Layer 3 of the cost model: the calibrated pruning decision (ADR-018).
//!
//! Autotuning measures the runtime of many candidate configs and keeps the fastest. Layer 2 ranks
//! them cheaply; Layer 3 decides **how much of that ranking to trust** — how aggressively to prune
//! before measuring — by calibrating the predicted order against a sample of real measurements.
//!
//! **Rank correlation, not Pearson (ADR-018).** Autotuning is a *ranking* task: what matters is that
//! the fast configs sort near the top (top-K recall), not that the predicted microseconds track the
//! measured ones linearly. A high Pearson correlation can coexist with a wrong top-K order, so the
//! gate uses **Spearman's rank correlation** (ADR-018; an earlier draft gated on Pearson, now
//! superseded — the schema field is `threshold_rank_correlation`).
//!
//! Three tiers, on the Spearman correlation between predicted and measured runtimes:
//! - **≥ 0.8** → `aggressive`: trust the ranking, measure only the top-K.
//! - **0.5–0.8** → `moderate`: partial trust, measure the top half.
//! - **< 0.5** → `disabled_fallback_full_sweep`: the kernel violates the model's assumptions
//!   (e.g. shared-memory bank conflicts the cost model can't see), so don't prune — measure all.

use cls_schema::PruningDecision;

/// Spearman correlation threshold at/above which pruning is aggressive.
const AGGRESSIVE_THRESHOLD: f64 = 0.8;
/// Threshold at/above which pruning is moderate; below it, pruning is disabled.
const MODERATE_THRESHOLD: f64 = 0.5;

/// How much of the autotune grid to prune, chosen by calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneMode {
    /// Rank correlation ≥ 0.8: measure only the predicted top-K.
    Aggressive,
    /// 0.5 ≤ rank correlation < 0.8: measure the top half.
    Moderate,
    /// Rank correlation < 0.5: don't prune — fall back to the full sweep.
    DisabledFallbackFullSweep,
}

impl PruneMode {
    /// The string the artifact records (`PruningDecision.mode`).
    pub fn as_str(self) -> &'static str {
        match self {
            PruneMode::Aggressive => "aggressive",
            PruneMode::Moderate => "moderate",
            PruneMode::DisabledFallbackFullSweep => "disabled_fallback_full_sweep",
        }
    }

    /// Whether this mode prunes at all (false only for the full-sweep fallback).
    pub fn prunes(self) -> bool {
        !matches!(self, PruneMode::DisabledFallbackFullSweep)
    }
}

/// Map a Spearman rank correlation to a prune mode (the three-tier gate).
pub fn prune_mode(rank_correlation: f64) -> PruneMode {
    if rank_correlation >= AGGRESSIVE_THRESHOLD {
        PruneMode::Aggressive
    } else if rank_correlation >= MODERATE_THRESHOLD {
        PruneMode::Moderate
    } else {
        PruneMode::DisabledFallbackFullSweep
    }
}

/// Spearman's rank correlation: Pearson correlation of the *ranks* of the two series (so it measures
/// monotonic agreement, robust to the nonlinear gap between predicted and measured times). Ties get
/// average ranks. `None` if the lengths differ, there are fewer than two points, or either series
/// has no rank variance (a constant series — correlation undefined).
pub fn spearman_correlation(predicted: &[f64], measured: &[f64]) -> Option<f64> {
    if predicted.len() != measured.len() || predicted.len() < 2 {
        return None;
    }
    pearson(&ranks(predicted), &ranks(measured))
}

/// Average ranks of the values (1-based; tied values share the mean of their positions).
fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![0.0; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        // positions i..j are tied; average their 1-based ranks.
        let avg_rank = ((i + 1 + j) as f64) / 2.0; // mean of (i+1 ..= j)
        for &k in &idx[i..j] {
            out[k] = avg_rank;
        }
        i = j;
    }
    out
}

/// Pearson correlation. `None` if either series has zero variance.
fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x - mean_a, y - mean_b);
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = (var_a * var_b).sqrt();
    (denom > 0.0).then_some(cov / denom)
}

/// The Layer-3 decision: calibrate the predicted runtimes against measured ones and decide how much
/// of a `configs_total`-config grid to measure, given the user's target `top_k`. `None` if the two
/// series can't be correlated (mismatched length, < 2 points, or constant).
///
/// `threshold_rank_correlation` records the Spearman correlation that drove the decision.
pub fn pruning_decision(
    predicted: &[f64],
    measured: &[f64],
    top_k: usize,
) -> Option<PruningDecision> {
    let rho = spearman_correlation(predicted, measured)?;
    let mode = prune_mode(rho);
    let configs_total = predicted.len() as u64;

    // How many configs to actually measure under each tier (the rest are pruned).
    let configs_measured = match mode {
        PruneMode::Aggressive => (top_k as u64).min(configs_total),
        PruneMode::Moderate => (top_k as u64)
            .max(configs_total.div_ceil(2))
            .min(configs_total),
        PruneMode::DisabledFallbackFullSweep => configs_total, // no pruning — measure everything
    };

    Some(PruningDecision {
        mode: Some(mode.as_str().to_string()),
        threshold_rank_correlation: Some(rho),
        configs_total: Some(configs_total),
        configs_predicted_top_k: Some(top_k as u64),
        configs_measured: Some(configs_measured),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_mode_has_three_tiers_at_the_thresholds() {
        assert_eq!(prune_mode(0.95), PruneMode::Aggressive);
        assert_eq!(prune_mode(0.8), PruneMode::Aggressive); // inclusive
        assert_eq!(prune_mode(0.65), PruneMode::Moderate);
        assert_eq!(prune_mode(0.5), PruneMode::Moderate); // inclusive
        assert_eq!(prune_mode(0.49), PruneMode::DisabledFallbackFullSweep);
        assert_eq!(prune_mode(-0.3), PruneMode::DisabledFallbackFullSweep);
    }

    #[test]
    fn spearman_is_one_for_a_perfectly_monotonic_pair() {
        // Predicted order matches measured order exactly (even though the values differ nonlinearly).
        let predicted = [1.0, 2.0, 3.0, 4.0];
        let measured = [10.0, 25.0, 26.0, 100.0];
        let rho = spearman_correlation(&predicted, &measured).unwrap();
        assert!((rho - 1.0).abs() < 1e-9, "got {rho}");
    }

    #[test]
    fn spearman_is_minus_one_for_a_reversed_pair() {
        let predicted = [1.0, 2.0, 3.0, 4.0];
        let measured = [4.0, 3.0, 2.0, 1.0];
        let rho = spearman_correlation(&predicted, &measured).unwrap();
        assert!((rho + 1.0).abs() < 1e-9, "got {rho}");
    }

    #[test]
    fn spearman_handles_ties_and_guards_degenerate_input() {
        // Tied predicted values -> average ranks; still a valid correlation.
        assert!(spearman_correlation(&[1.0, 1.0, 2.0], &[5.0, 6.0, 7.0]).is_some());
        // A constant series has no rank variance -> undefined.
        assert_eq!(
            spearman_correlation(&[3.0, 3.0, 3.0], &[1.0, 2.0, 3.0]),
            None
        );
        // Length mismatch / too few points.
        assert_eq!(spearman_correlation(&[1.0, 2.0], &[1.0]), None);
        assert_eq!(spearman_correlation(&[1.0], &[1.0]), None);
    }

    #[test]
    fn well_calibrated_grid_prunes_aggressively_to_top_k() {
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let measured = [1.1, 2.2, 2.9, 4.3, 5.1, 6.0, 7.2, 8.1]; // monotonic with predicted
        let d = pruning_decision(&predicted, &measured, 2).unwrap();
        assert_eq!(d.mode.as_deref(), Some("aggressive"));
        assert_eq!(d.configs_total, Some(8));
        assert_eq!(d.configs_measured, Some(2)); // only the top-K
        assert!(d.threshold_rank_correlation.unwrap() >= 0.8);
    }

    #[test]
    fn bank_conflict_kernel_falls_back_to_full_sweep() {
        // Worked-example failure: a shared-memory / bank-conflict kernel violates the model's
        // assumptions, so the predicted order disagrees with the measured order (here, reversed).
        // Rank correlation < 0.5 -> pruning disabled, full sweep.
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let measured = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let d = pruning_decision(&predicted, &measured, 2).unwrap();
        assert_eq!(d.mode.as_deref(), Some("disabled_fallback_full_sweep"));
        assert_eq!(d.configs_measured, Some(6)); // measure everything — no pruning
        assert!(d.threshold_rank_correlation.unwrap() < 0.5);
    }

    #[test]
    fn moderate_calibration_measures_the_top_half() {
        // A predicted order partly right, partly wrong -> Spearman in the 0.5-0.8 band.
        // (Sum of squared rank diffs = 12, so rho = 1 - 6·12/(6·35) ≈ 0.657.)
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let measured = [3.0, 1.0, 2.0, 6.0, 4.0, 5.0];
        let rho = spearman_correlation(&predicted, &measured).unwrap();
        assert!(
            (MODERATE_THRESHOLD..AGGRESSIVE_THRESHOLD).contains(&rho),
            "rho={rho}"
        );
        let d = pruning_decision(&predicted, &measured, 1).unwrap();
        assert_eq!(d.mode.as_deref(), Some("moderate"));
        assert_eq!(d.configs_measured, Some(3)); // top half of 6
    }
}
