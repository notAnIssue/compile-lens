//! Kernel metadata (`kernels[]`) and the three-layer roofline model
//! (`roofline_predictions[]`).
//!
//! Like [`crate::records`], these structs accept-but-do-not-preserve unknown keys (no
//! `extra` capture); see the crate-level docs.

use serde::{Deserialize, Serialize};

use crate::JsonMap;

/// Triton launch configuration. `grid`/`block` are kept as `Vec<u64>` (not `[u64; 3]`):
/// shape enforcement (exactly 3 dims) is the JSON-Schema validator's job, so a malformed
/// producer yields a validation error rather than a hard serde panic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaunchConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grid: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub block: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mem_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_warps: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_stages: Option<u64>,
}

/// Kernel features used by the roofline model. `flops` / `bytes_*` are `f64` because the
/// schema types them as `number` (they routinely arrive in scientific notation, e.g.
/// `1.2e10`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelFeatures {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flops: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_loaded: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_stored: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<u64>,
    /// Register pressure proxy (Tool 5's 4th correction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_regs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_spills: Option<u64>,
}

/// Measured kernel timings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelMeasurements {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_us: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_us: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99_us: Option<f64>,
    /// Data source: `proton` or `self_timed`. Kept `String` (not enum) for the same
    /// forward-compat reason as the lint vocabularies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A generated Triton kernel and its measurements (`kernels[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kernel {
    pub kernel_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ptx_path: Option<String>,
    /// Opt-in only; omitted by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_source_excerpt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch_config: Option<LaunchConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occupancy_hint: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<KernelFeatures>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurements: Option<KernelMeasurements>,
    /// Free-form per-config autotune records; left as opaque JSON objects for v0.5.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub autotune_history: Vec<JsonMap>,
}

/// GPU peak-rate spec used to place a kernel on the roofline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_compute_tflops_bf16: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_bw_gbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l2_cache_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_count: Option<u64>,
}

/// Calibration of the empirical predictor against measured runtimes. `*_correlation` are
/// `f64` (not unsigned) since correlations may be negative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pearson_correlation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spearman_correlation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_verdict: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_samples: Option<u64>,
}

/// Autotune pruning decision (`safe_prune_enabled` / `partial_prune` /
/// `disabled_fallback_full_sweep`). `mode` kept `String` for forward compat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PruningDecision {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_pearson: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configs_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configs_predicted_top_k: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configs_measured: Option<u64>,
}

/// An evidence-linked optimization hint (never an auto-applied patch, N2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_link: Option<String>,
}

/// The three-layer roofline prediction for one kernel (`roofline_predictions[]`):
/// theoretical lower bound / empirical predictor / measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RooflinePrediction {
    /// Cross-reference into `kernels[]`.
    pub kernel_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_spec: Option<GpuSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arithmetic_intensity: Option<f64>,
    /// `memory` or `compute`. Kept `String` for forward compat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_type: Option<String>,
    /// Ideal Williams roofline; reference only, not used for pruning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theoretical_lower_bound_us: Option<f64>,
    /// After 4 corrections; used for ranking/pruning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empirical_predictor_us: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_us: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub achieved_vs_predictor: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections_applied: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruning_decision: Option<PruningDecision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<Suggestion>,
}
