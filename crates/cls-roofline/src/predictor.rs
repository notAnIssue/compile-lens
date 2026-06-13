//! Layer 2 of the cost model: the empirical runtime predictor (design.md §8.5).
//!
//! Layer 1's lower bound is the *ideal* floor; real kernels pay more. Layer 2 inflates it with four
//! corrections into a predictor used for **ranking and pruning** (Layer 3), not for absolute
//! accuracy — the contract is monotonic ranking, validated by rank correlation, not µs precision.
//!
//! ```text
//! empirical_us = lower_bound_us × (1 + block_size_penalty) × (1 + occupancy_penalty)
//!                + launch_overhead_us + register_pressure_penalty_us
//! ```
//!
//! 1. **block_size** — small blocks underutilize the memory pipeline; a ramp below a saturation
//!    threshold. (The design's literal `block_size × num_warps × dtype_bytes < L2_line` predicate is
//!    dimensionally ambiguous, so this implements its *intent* — a small-block bandwidth
//!    under-utilization penalty — as a documented threshold ramp.)
//! 2. **occupancy** — register-limited occupancy from the NVIDIA occupancy method; low occupancy
//!    can't hide latency, so the penalty is the inverse-occupancy shortfall, capped.
//! 3. **launch_overhead** — a fixed additive per-launch cost; dominates for tiny kernels.
//! 4. **register_pressure** — spilling adds local-memory traffic. **Softened (ADR-018):** spill goes
//!    to *local memory*, usually L1/L2-cached, reaching HBM only on a miss — so it is a *modest*
//!    amplifier on register-bound kernels, not the "doubles bandwidth / biggest factor on H100"
//!    claim an earlier draft made. Capped well below 2×.
//!
//! The coefficients are documented defaults, calibration-tunable; the model earns trust through
//! Layer 3's rank correlation against real measurements, not through these constants being exact.

use cls_schema::KernelFeatures;

use crate::roofline::RooflineCostModel;

// ── tunable coefficients (documented defaults) ──────────────────────────────────────────

/// Threads-per-block at/above which the memory pipeline is considered saturated (4 warps). Below
/// it, the block_size penalty ramps up.
const BLOCK_SATURATION_THREADS: f64 = 128.0;
/// Maximum block_size penalty (for a degenerate single-warp block).
const BLOCK_PENALTY_MAX: f64 = 0.5;

/// Registers per SM (Ampere / Hopper / Blackwell are all 65536). A per-arch table is a later
/// refinement; this holds across the registry's current GPUs.
const REGS_PER_SM: f64 = 65536.0;
/// Maximum resident warps per SM (64 on Ampere / Hopper / Blackwell).
const MAX_WARPS_PER_SM: f64 = 64.0;
const WARP_SIZE: f64 = 32.0;
/// Cap on the occupancy penalty, so a near-zero occupancy estimate can't blow the prediction up
/// without bound (an occupancy of ~0.33 already saturates it).
const OCCUPANCY_PENALTY_MAX: f64 = 2.0;

/// Per-launch overhead in microseconds (design.md §8.5: ~5 µs). A documented default until
/// calibrated against real measurements.
pub const LAUNCH_OVERHEAD_US: f64 = 5.0;

/// Cap on the register-pressure penalty as a fraction of the lower bound — the softening: even an
/// all-spill kernel adds at most +50 %, not the 2× an earlier draft claimed.
const SPILL_PENALTY_CAP: f64 = 0.5;

/// The Layer-2 result: the predicted runtime and which corrections were active.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalPrediction {
    pub predicted_us: f64,
    /// Names of the corrections that contributed (`launch_overhead` is always present).
    pub corrections_applied: Vec<String>,
}

/// Block-size penalty: a ramp that is zero once the block saturates the pipeline
/// (`>= BLOCK_SATURATION_THREADS`) and rises toward [`BLOCK_PENALTY_MAX`] for tiny blocks. Unknown
/// block size → no penalty.
pub fn block_size_penalty(block_size: Option<u64>) -> f64 {
    match block_size {
        Some(bs) if (bs as f64) < BLOCK_SATURATION_THREADS => {
            let shortfall = (BLOCK_SATURATION_THREADS - bs as f64) / BLOCK_SATURATION_THREADS;
            shortfall * BLOCK_PENALTY_MAX
        }
        _ => 0.0,
    }
}

/// Register-limited occupancy in `[0, 1]` via the NVIDIA occupancy method: how many blocks fit per
/// SM under the register budget, times warps per block, against the SM's warp ceiling. `None` if
/// `num_regs` or `block_size` is unknown (occupancy can't be estimated).
pub fn occupancy(block_size: Option<u64>, num_regs: Option<u64>) -> Option<f64> {
    let bs = block_size? as f64;
    let regs = num_regs? as f64;
    if bs <= 0.0 || regs <= 0.0 {
        return None;
    }
    let regs_per_block = regs * bs;
    let blocks_per_sm = (REGS_PER_SM / regs_per_block).floor();
    let warps_per_block = (bs / WARP_SIZE).ceil();
    let active_warps = (blocks_per_sm * warps_per_block).min(MAX_WARPS_PER_SM);
    Some((active_warps / MAX_WARPS_PER_SM).clamp(0.0, 1.0))
}

/// Occupancy penalty: the inverse-occupancy shortfall `(1/occ − 1)`, capped. Full occupancy → 0;
/// unknown occupancy → 0 (no penalty rather than a guess).
pub fn occupancy_penalty(occupancy: Option<f64>) -> f64 {
    match occupancy {
        Some(occ) if occ > 0.0 => (1.0 / occ - 1.0).min(OCCUPANCY_PENALTY_MAX),
        _ => 0.0,
    }
}

/// Register-pressure penalty in microseconds (additive, **softened**). Zero without spills; with
/// spills, a fraction of the lower bound proportional to how much of the register traffic spills
/// (`n_spills / (num_regs + n_spills)`), capped at [`SPILL_PENALTY_CAP`]. Without `num_regs`, a
/// spill is treated as a small flat fraction.
pub fn register_pressure_penalty_us(
    lower_bound_us: f64,
    num_regs: Option<u64>,
    n_spills: Option<u64>,
) -> f64 {
    let spills = n_spills.unwrap_or(0) as f64;
    if spills <= 0.0 {
        return 0.0;
    }
    let severity = match num_regs {
        Some(regs) if regs > 0 => spills / (regs as f64 + spills),
        _ => 0.1, // spills present but register count unknown: a small flat amplifier
    };
    lower_bound_us * severity.min(SPILL_PENALTY_CAP)
}

impl RooflineCostModel {
    /// Apply the four Layer-2 corrections to a Layer-1 lower bound, producing the empirical
    /// predictor and the list of corrections that contributed.
    pub fn empirical_predictor(
        &self,
        lower_bound_us: f64,
        features: &KernelFeatures,
    ) -> EmpiricalPrediction {
        let bs_pen = block_size_penalty(features.block_size);
        let occ = occupancy(features.block_size, features.num_regs);
        let occ_pen = occupancy_penalty(occ);
        let reg_pen_us =
            register_pressure_penalty_us(lower_bound_us, features.num_regs, features.n_spills);

        let predicted_us =
            lower_bound_us * (1.0 + bs_pen) * (1.0 + occ_pen) + LAUNCH_OVERHEAD_US + reg_pen_us;

        let mut corrections_applied = Vec::new();
        if bs_pen > 0.0 {
            corrections_applied.push("block_size".to_string());
        }
        if occ_pen > 0.0 {
            corrections_applied.push("occupancy".to_string());
        }
        corrections_applied.push("launch_overhead".to_string()); // always added
        if reg_pen_us > 0.0 {
            corrections_applied.push("register_pressure".to_string());
        }

        EmpiricalPrediction {
            predicted_us,
            corrections_applied,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a100() -> RooflineCostModel {
        RooflineCostModel::for_gpu("A100-SXM-80GB").expect("registered")
    }

    fn features(
        block_size: Option<u64>,
        num_regs: Option<u64>,
        n_spills: Option<u64>,
    ) -> KernelFeatures {
        KernelFeatures {
            flops: Some(1.0e12),
            bytes_loaded: Some(1.0e10),
            bytes_stored: Some(0.0),
            block_size,
            num_regs,
            n_spills,
        }
    }

    #[test]
    fn block_size_penalty_is_zero_when_saturated_and_ramps_below() {
        assert_eq!(block_size_penalty(Some(256)), 0.0); // >= 128 saturates
        assert_eq!(block_size_penalty(Some(128)), 0.0);
        assert!(block_size_penalty(Some(64)) > 0.0);
        assert!(block_size_penalty(Some(32)) > block_size_penalty(Some(64))); // smaller is worse
        assert_eq!(block_size_penalty(None), 0.0); // unknown -> no penalty
    }

    #[test]
    fn occupancy_is_register_limited() {
        // 32 regs/thread × 256 threads = 8192 regs/block; 65536/8192 = 8 blocks/SM;
        // 8 warps/block × 8 = 64 warps = full occupancy (1.0).
        assert_eq!(occupancy(Some(256), Some(32)), Some(1.0));
        // 128 regs/thread is heavy: 128 × 256 = 32768; 65536/32768 = 2 blocks/SM;
        // 8 warps/block × 2 = 16 warps / 64 = 0.25 occupancy.
        assert_eq!(occupancy(Some(256), Some(128)), Some(0.25));
        assert_eq!(occupancy(Some(256), None), None); // unknown regs
    }

    #[test]
    fn occupancy_penalty_is_inverse_shortfall_capped() {
        assert_eq!(occupancy_penalty(Some(1.0)), 0.0); // full occupancy, no penalty
        assert!((occupancy_penalty(Some(0.5)) - 1.0).abs() < 1e-9); // 1/0.5 - 1 = 1.0
        assert_eq!(occupancy_penalty(Some(0.1)), OCCUPANCY_PENALTY_MAX); // capped
        assert_eq!(occupancy_penalty(None), 0.0);
    }

    #[test]
    fn register_pressure_penalty_is_softened_and_capped() {
        // No spills -> no penalty.
        assert_eq!(register_pressure_penalty_us(100.0, Some(64), Some(0)), 0.0);
        // Some spills: a fraction of the lower bound, well under 2×.
        let p = register_pressure_penalty_us(100.0, Some(64), Some(16));
        assert!(p > 0.0 && p < 100.0 * SPILL_PENALTY_CAP + 1e-9);
        // Heavy spilling is capped at +50% of the lower bound (the softening — not 2×).
        let heavy = register_pressure_penalty_us(100.0, Some(8), Some(1000));
        assert!((heavy - 100.0 * SPILL_PENALTY_CAP).abs() < 1e-9);
    }

    #[test]
    fn launch_overhead_is_always_applied() {
        // A well-shaped kernel (saturated block, full occupancy, no spills): only launch overhead.
        let pred = a100().empirical_predictor(50.0, &features(Some(256), Some(32), Some(0)));
        assert_eq!(
            pred.corrections_applied,
            vec!["launch_overhead".to_string()]
        );
        assert!((pred.predicted_us - (50.0 + LAUNCH_OVERHEAD_US)).abs() < 1e-9);
    }

    #[test]
    fn predictor_is_at_least_the_lower_bound_plus_overhead() {
        // Corrections only ever inflate, never shrink, the lower bound.
        let pred = a100().empirical_predictor(50.0, &features(Some(64), Some(128), Some(8)));
        assert!(pred.predicted_us >= 50.0 + LAUNCH_OVERHEAD_US);
    }

    #[test]
    fn high_spill_kernel_triggers_register_pressure_correction() {
        // A synthetic high-spill kernel: register_pressure must be among the applied corrections.
        let pred = a100().empirical_predictor(100.0, &features(Some(256), Some(40), Some(80)));
        assert!(pred
            .corrections_applied
            .contains(&"register_pressure".to_string()));
    }
}
