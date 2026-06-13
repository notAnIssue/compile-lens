//! The GPU spec registry: peak compute and memory bandwidth for the roofline model.
//!
//! These are the published dense **BF16 Tensor-Core** peak and peak HBM bandwidth — the two
//! quantities a Williams roofline needs. The roofline's baseline precision is BF16 (the inference
//! default); a kernel in another dtype would need a different compute peak, which is a documented v0
//! assumption rather than a per-dtype table.
//!
//! Seeded with the datacenter inference baseline (A100 / H100 / B200). Consumer / workstation cards
//! (e.g. RTX 4090, L40) are deferred until their BF16 rate is pinned precisely — those parts gate
//! FP16-accumulate matmuls at half the FP16-accumulate rate, so a single "peak BF16" number is
//! ambiguous and must be chosen deliberately rather than guessed. Adding a GPU is a data edit here.

/// A GPU's roofline-relevant specification. `peak_compute_tflops_bf16` and `peak_bw_gbps` are
/// required (the roofline cannot be drawn without them); `l2_cache_mb` / `sm_count` are optional
/// secondary metadata used by later correction layers.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuSpec {
    /// Canonical name, matched case-insensitively by [`lookup`].
    pub name: &'static str,
    /// Dense BF16 Tensor-Core peak, in TFLOP/s (not the with-sparsity figure).
    pub peak_compute_tflops_bf16: f64,
    /// Peak memory (HBM) bandwidth, in GB/s.
    pub peak_bw_gbps: f64,
    /// L2 cache size in MB, when known.
    pub l2_cache_mb: Option<f64>,
    /// Streaming-multiprocessor count, when known.
    pub sm_count: Option<u64>,
}

impl GpuSpec {
    /// Peak compute in FLOP/s.
    pub fn peak_compute_flops(&self) -> f64 {
        self.peak_compute_tflops_bf16 * 1e12
    }

    /// Peak memory bandwidth in bytes/s.
    pub fn peak_bw_bytes_per_s(&self) -> f64 {
        self.peak_bw_gbps * 1e9
    }

    /// The serialization-facing [`cls_schema::GpuSpec`] (all fields wrapped, for the artifact).
    pub fn to_schema(&self) -> cls_schema::GpuSpec {
        cls_schema::GpuSpec {
            name: Some(self.name.to_string()),
            peak_compute_tflops_bf16: Some(self.peak_compute_tflops_bf16),
            peak_bw_gbps: Some(self.peak_bw_gbps),
            l2_cache_mb: self.l2_cache_mb,
            sm_count: self.sm_count,
        }
    }
}

/// A100 SXM (80 GB): 312 TFLOP/s dense BF16, 2039 GB/s HBM2e, 40 MB L2, 108 SMs.
pub const A100_SXM_80GB: GpuSpec = GpuSpec {
    name: "A100-SXM-80GB",
    peak_compute_tflops_bf16: 312.0,
    peak_bw_gbps: 2039.0,
    l2_cache_mb: Some(40.0),
    sm_count: Some(108),
};

/// H100 SXM: 989.5 TFLOP/s dense BF16 (1979 with sparsity), 3350 GB/s HBM3, 50 MB L2, 132 SMs.
pub const H100_SXM: GpuSpec = GpuSpec {
    name: "H100-SXM",
    peak_compute_tflops_bf16: 989.5,
    peak_bw_gbps: 3350.0,
    l2_cache_mb: Some(50.0),
    sm_count: Some(132),
};

/// B200 (Blackwell): ~2250 TFLOP/s dense BF16, 8000 GB/s HBM3e. L2 / SM count left unset — the
/// dual-reticle design does not publish them as a single comparable figure, so they are omitted
/// rather than guessed.
pub const B200: GpuSpec = GpuSpec {
    name: "B200",
    peak_compute_tflops_bf16: 2250.0,
    peak_bw_gbps: 8000.0,
    l2_cache_mb: None,
    sm_count: None,
};

/// The registry, in a stable order.
pub const REGISTRY: &[GpuSpec] = &[A100_SXM_80GB, H100_SXM, B200];

/// Look a GPU up by name, case-insensitively. Returns a reference into the static registry.
pub fn lookup(name: &str) -> Option<&'static GpuSpec> {
    REGISTRY.iter().find(|g| g.name.eq_ignore_ascii_case(name))
}

/// The names of all registered GPUs, for error messages and `--list`-style help.
pub fn known_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|g| g.name).collect()
}
