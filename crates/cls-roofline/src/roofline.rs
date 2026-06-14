//! Layer 1 of the three-layer cost model: the Williams roofline — a *theoretical lower bound* on a
//! kernel's runtime from its arithmetic intensity and the GPU's peak compute and bandwidth.
//!
//! For flops `F` and moved bytes `B` on a GPU with peak compute `C` (FLOP/s) and peak bandwidth `M`
//! (bytes/s):
//!
//! - **arithmetic intensity** `AI = F / B` (FLOP per byte).
//! - **ridge point** `AI* = C / M` — the intensity where the compute and bandwidth ceilings meet.
//!   Below it a kernel is *memory-bound*; at or above it, *compute-bound*.
//! - **attainable throughput** `min(C, M · AI)` (the roofline itself).
//! - **theoretical lower-bound time** `max(F / C, B / M)` — whichever of the compute-limited and
//!   bandwidth-limited times is larger. No kernel can beat this on this GPU; it is Layer 1's output,
//!   the floor the empirical predictor (Layer 2) and pruning (Layer 3) build on.
//!
//! Pure arithmetic — no GPU, no measurement. Layers 2 and 3 land separately; this fills the Layer-1
//! fields of [`cls_schema::RooflinePrediction`] and leaves the rest unset.

use cls_schema::{KernelFeatures, RooflinePrediction};

use crate::gpu::{self, GpuSpec};

/// Whether a kernel is limited by memory bandwidth or by compute, at its arithmetic intensity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundType {
    /// `AI < ridge`: bandwidth is the binding ceiling.
    Memory,
    /// `AI >= ridge`: compute is the binding ceiling.
    Compute,
}

impl BoundType {
    /// The string the artifact records (`RooflinePrediction.bound_type`: `memory` / `compute`).
    pub fn as_str(self) -> &'static str {
        match self {
            BoundType::Memory => "memory",
            BoundType::Compute => "compute",
        }
    }
}

/// The roofline cost model bound to one GPU. `for_gpu` is the stable entry point downstream tools
/// build on.
#[derive(Debug, Clone)]
pub struct RooflineCostModel {
    gpu: &'static GpuSpec,
}

impl RooflineCostModel {
    /// Construct a model for a registered GPU, by name (case-insensitive). `None` if unknown.
    pub fn for_gpu(name: &str) -> Option<Self> {
        gpu::lookup(name).map(|gpu| Self { gpu })
    }

    /// The GPU this model is bound to.
    pub fn gpu(&self) -> &GpuSpec {
        self.gpu
    }

    /// Arithmetic intensity `F / B` (FLOP per byte). `None` if `bytes` is zero (intensity undefined
    /// — there is no memory traffic to divide by).
    pub fn arithmetic_intensity(flops: f64, bytes: f64) -> Option<f64> {
        (bytes > 0.0).then_some(flops / bytes)
    }

    /// The ridge-point intensity `C / M` (FLOP per byte) where the two ceilings cross.
    pub fn ridge_point(&self) -> f64 {
        self.gpu.peak_compute_flops() / self.gpu.peak_bw_bytes_per_s()
    }

    /// The attainable throughput `min(C, M · AI)` in FLOP/s at a given arithmetic intensity.
    pub fn attainable_flops(&self, arithmetic_intensity: f64) -> f64 {
        let bw_bound = self.gpu.peak_bw_bytes_per_s() * arithmetic_intensity;
        bw_bound.min(self.gpu.peak_compute_flops())
    }

    /// Which ceiling binds at this arithmetic intensity. At exactly the ridge point the kernel is
    /// compute-bound (the ceilings are equal; the convention picks compute).
    pub fn bound_type(&self, arithmetic_intensity: f64) -> BoundType {
        if arithmetic_intensity < self.ridge_point() {
            BoundType::Memory
        } else {
            BoundType::Compute
        }
    }

    /// The theoretical lower-bound runtime in microseconds: `max(F / C, B / M)`.
    pub fn theoretical_lower_bound_us(&self, flops: f64, bytes: f64) -> f64 {
        let compute_s = flops / self.gpu.peak_compute_flops();
        let bw_s = bytes / self.gpu.peak_bw_bytes_per_s();
        compute_s.max(bw_s) * 1e6
    }

    /// Run Layer 1 over a kernel's features, producing a [`RooflinePrediction`] with the Layer-1
    /// fields filled (arithmetic intensity, bound type, theoretical lower bound, the GPU spec). The
    /// empirical-predictor, measurement, calibration, and pruning fields are left for later layers.
    ///
    /// Returns `None` if `flops` or `bytes_*` are missing — Layer 1 needs both to draw the roofline.
    pub fn analyze(
        &self,
        kernel_id: &str,
        features: &KernelFeatures,
    ) -> Option<RooflinePrediction> {
        let flops = features.flops?;
        let bytes = features.bytes_loaded? + features.bytes_stored.unwrap_or(0.0);
        let ai = Self::arithmetic_intensity(flops, bytes)?;
        Some(RooflinePrediction {
            kernel_id: kernel_id.to_string(),
            gpu_spec: Some(self.gpu.to_schema()),
            arithmetic_intensity: Some(ai),
            bound_type: Some(self.bound_type(ai).as_str().to_string()),
            theoretical_lower_bound_us: Some(self.theoretical_lower_bound_us(flops, bytes)),
            ..default_prediction(kernel_id)
        })
    }

    /// Full single-kernel prediction: Layer 1 (the theoretical roofline) plus Layer 2 (the
    /// empirical predictor's four corrections). On top of [`analyze`](Self::analyze)'s Layer-1
    /// fields it fills `empirical_predictor_us` (the number Layer 3 ranks and prunes on) and the
    /// `corrections_applied` list. `None` when Layer 1 can't draw the roofline (missing
    /// flops/bytes).
    pub fn predict(
        &self,
        kernel_id: &str,
        features: &KernelFeatures,
    ) -> Option<RooflinePrediction> {
        let mut prediction = self.analyze(kernel_id, features)?;
        // `analyze` returning `Some` guarantees the lower bound is set.
        let lower_bound_us = prediction.theoretical_lower_bound_us?;
        let empirical = self.empirical_predictor(lower_bound_us, features);
        prediction.empirical_predictor_us = Some(empirical.predicted_us);
        prediction.corrections_applied = empirical.corrections_applied;
        Some(prediction)
    }
}

/// A `RooflinePrediction` with only `kernel_id` set and every analysis field unset — the base the
/// layers fill in. (`RooflinePrediction` has no `Default` because `kernel_id` is required.)
fn default_prediction(kernel_id: &str) -> RooflinePrediction {
    RooflinePrediction {
        kernel_id: kernel_id.to_string(),
        gpu_spec: None,
        arithmetic_intensity: None,
        bound_type: None,
        theoretical_lower_bound_us: None,
        empirical_predictor_us: None,
        measured_us: None,
        achieved_vs_predictor: None,
        corrections_applied: Vec::new(),
        calibration: None,
        pruning_decision: None,
        verdict: None,
        suggestions: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a100() -> RooflineCostModel {
        RooflineCostModel::for_gpu("A100-SXM-80GB").expect("A100 is registered")
    }

    #[test]
    fn for_gpu_is_case_insensitive_and_rejects_unknown() {
        assert!(RooflineCostModel::for_gpu("a100-sxm-80gb").is_some());
        assert!(RooflineCostModel::for_gpu("H100-SXM").is_some());
        assert!(RooflineCostModel::for_gpu("no-such-gpu").is_none());
    }

    #[test]
    fn ridge_point_matches_peak_compute_over_bandwidth() {
        // A100: 312e12 / 2039e9 ≈ 153 FLOP/byte.
        let ridge = a100().ridge_point();
        assert!((ridge - 153.0).abs() < 1.0, "A100 ridge ~153, got {ridge}");
    }

    #[test]
    fn arithmetic_intensity_is_flops_over_bytes() {
        assert_eq!(
            RooflineCostModel::arithmetic_intensity(1.0e9, 1.0e8),
            Some(10.0)
        );
        // Undefined with no memory traffic.
        assert_eq!(RooflineCostModel::arithmetic_intensity(1.0e9, 0.0), None);
    }

    #[test]
    fn bound_type_splits_at_the_ridge() {
        let m = a100();
        let ridge = m.ridge_point();
        assert_eq!(m.bound_type(ridge - 1.0), BoundType::Memory);
        assert_eq!(m.bound_type(ridge + 1.0), BoundType::Compute);
        // At exactly the ridge, the convention is compute-bound.
        assert_eq!(m.bound_type(ridge), BoundType::Compute);
    }

    #[test]
    fn attainable_throughput_is_the_lower_ceiling() {
        let m = a100();
        // Memory-bound (AI=10 < ridge): bandwidth ceiling = 2039e9 * 10.
        assert!((m.attainable_flops(10.0) - 2039e9 * 10.0).abs() < 1e6);
        // Compute-bound (AI=1000 > ridge): clamped to peak compute.
        assert!((m.attainable_flops(1000.0) - 312e12).abs() < 1e6);
    }

    #[test]
    fn lower_bound_is_the_larger_of_compute_and_bandwidth_time() {
        let m = a100();
        // Memory-bound kernel: bandwidth time dominates. B/M = 2.039e12 / 2.039e12 = 1 s = 1e6 us.
        let memory_bound = m.theoretical_lower_bound_us(1.0e9, 2.039e12);
        assert!((memory_bound - 1.0e6).abs() < 1.0, "got {memory_bound}");
        // Compute-bound kernel: compute time dominates. F/C = 3.12e14 / 3.12e14 = 1 s = 1e6 us.
        let compute_bound = m.theoretical_lower_bound_us(3.12e14, 1.0e3);
        assert!((compute_bound - 1.0e6).abs() < 1.0, "got {compute_bound}");
    }

    #[test]
    fn analyze_fills_layer1_fields_and_leaves_the_rest_unset() {
        let features = KernelFeatures {
            flops: Some(2.0e12),
            bytes_loaded: Some(1.0e10),
            bytes_stored: Some(1.0e10),
            block_size: None,
            num_regs: None,
            n_spills: None,
        };
        let pred = a100()
            .analyze("k0", &features)
            .expect("flops + bytes present");
        assert_eq!(pred.kernel_id, "k0");
        assert_eq!(pred.arithmetic_intensity, Some(2.0e12 / 2.0e10)); // 100 FLOP/byte
        assert_eq!(pred.bound_type.as_deref(), Some("memory")); // 100 < 153 ridge
        assert!(pred.theoretical_lower_bound_us.is_some());
        assert_eq!(
            pred.gpu_spec.and_then(|g| g.name).as_deref(),
            Some("A100-SXM-80GB")
        );
        // Later layers' fields stay unset.
        assert!(pred.empirical_predictor_us.is_none());
        assert!(pred.pruning_decision.is_none());
        assert!(pred.corrections_applied.is_empty());
    }

    #[test]
    fn analyze_needs_both_flops_and_bytes() {
        let no_flops = KernelFeatures {
            flops: None,
            bytes_loaded: Some(1.0e10),
            bytes_stored: None,
            block_size: None,
            num_regs: None,
            n_spills: None,
        };
        assert!(a100().analyze("k", &no_flops).is_none());
    }

    #[test]
    fn predict_composes_layer1_and_layer2() {
        // A small-block, register-heavy, spilling kernel: Layer 2 must inflate Layer 1's bound.
        let features = KernelFeatures {
            flops: Some(1.0e12),
            bytes_loaded: Some(1.0e10),
            bytes_stored: Some(0.0),
            block_size: Some(64), // below saturation -> block_size penalty
            num_regs: Some(128),  // heavy regs -> low occupancy -> occupancy penalty
            n_spills: Some(16),   // spills -> register_pressure penalty
        };
        let pred = a100()
            .predict("k0", &features)
            .expect("flops + bytes present");
        // Layer 1 fields are present...
        let lb = pred
            .theoretical_lower_bound_us
            .expect("layer 1 lower bound");
        assert!(pred.arithmetic_intensity.is_some());
        // ...and Layer 2 filled the empirical predictor strictly above the lower bound.
        let emp = pred
            .empirical_predictor_us
            .expect("layer 2 empirical predictor");
        assert!(emp > lb, "empirical {emp} must exceed lower bound {lb}");
        // All three feature-driven corrections fired, plus the always-on launch overhead.
        for correction in [
            "block_size",
            "occupancy",
            "launch_overhead",
            "register_pressure",
        ] {
            assert!(
                pred.corrections_applied.iter().any(|c| c == correction),
                "expected correction {correction} in {:?}",
                pred.corrections_applied
            );
        }
    }

    #[test]
    fn predict_is_none_without_the_roofline_inputs() {
        let no_bytes = KernelFeatures {
            flops: Some(1.0e12),
            bytes_loaded: None,
            bytes_stored: None,
            block_size: None,
            num_regs: None,
            n_spills: None,
        };
        assert!(a100().predict("k", &no_bytes).is_none());
    }
}
