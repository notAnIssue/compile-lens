//! `cls-schema` — Rust bindings for the compile-lens `.cls.json` session artifact
//! (schema **v0.5.0**).
//!
//! This crate is the Rust half of the *sole* cross-language contract between the Python
//! collectors (which write `.cls.json`) and the Rust analyzers (which read it). There is
//! no PyO3 and no IPC — the file on disk is the API (the design doc). The types here
//! mirror [`schema/v0.5.0.json`] 1:1.
//!
//! See ADR-021 for the *layout* decision (semantic-domain hybrid; nest objects, flatten
//! scalars; normalized top-level arrays). The serde-representation rationale lives in the
//! per-decision comments below.
//!
//! # Representation conventions
//!
//! - **Number mapping.** schema `integer` → [`u64`], schema `number` → [`f64`], schema
//!   `string` → [`String`]. One uniform rule, no per-field micro-typing.
//! - **Optional vs. nullable.** A field that is merely *absent when unknown* is
//!   `Option<T>` and skipped on serialize (so a minimal artifact stays minimal). The two
//!   schema `[T, null]` *union* fields — [`CompileConfig::dynamic`] and
//!   [`EnvSnapshot::random_seed`] — are always serialized, because their `null` state is
//!   meaningful (torch.compile semantics) and must survive a round trip.
//! - **Deterministic maps.** Open maps use [`indexmap::IndexMap`], not `HashMap`, to keep
//!   iteration order stable across runs (D10 reproducibility).
//! - **Forward compatibility.** `additionalProperties: true` holds at every level of the
//!   schema. We preserve unknown keys via a flattened [`JsonMap`] *only at the two growth
//!   points* — [`ClsArtifact`] (new top-level arrays) and [`Session`] (D6 says session
//!   metadata only grows: `gpu_arch`, `driver_version`, `nccl_version`, …). Deeper record
//!   structs accept-and-ignore unknown keys instead (see design note for why this line is
//!   drawn here).
//!
//! [`schema/v0.5.0.json`]: https://compile-lens.dev/schema/v0.5.0.json

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

mod kernels;
mod records;
mod session;

pub use kernels::{
    Calibration, GpuSpec, Kernel, KernelFeatures, KernelMeasurements, LaunchConfig,
    PruningDecision, RooflinePrediction, Suggestion,
};
pub use records::{
    CompilePhase, CompiledGraph, FailedGuard, FxNode, GraphBreak, GuardEvaluation,
    InternalStateSnapshot, Iteration, LintFinding, Recompilation, ReferenceIssue, SourceLocation,
    SourceRange,
};
pub use session::{CompileConfig, EnvSnapshot, RedactionPolicy, Session};

/// Schema version these types mirror.
pub const SCHEMA_VERSION: &str = "0.5.0";

/// A `string -> string` map with deterministic iteration order (D10).
///
/// Used for closed-value-type maps such as `collector_versions` and
/// `env_snapshot.relevant_env_vars`.
pub type StringMap = IndexMap<String, String>;

/// A `string -> arbitrary-JSON` map with deterministic iteration order (D10).
///
/// Used for open maps (`compile_config.options`, `compile_phases_summary`) and for the
/// flattened `extra` capture of unknown keys.
pub type JsonMap = IndexMap<String, serde_json::Value>;

/// Top-level `.cls.json` artifact: a `session` plus the normalized parallel record arrays
/// (ADR-021). Records cross-reference each other by `*_id` rather than nesting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClsArtifact {
    /// Strict SemVer; a minor bump requires a `cls-schema-migrate` function (D10).
    pub schema_version: String,
    /// Run-level metadata.
    pub session: Session,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub graph_breaks: Vec<GraphBreak>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recompilations: Vec<Recompilation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compiled_graphs: Vec<CompiledGraph>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compile_phases: Vec<CompilePhase>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub iterations: Vec<Iteration>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lint_findings: Vec<LintFinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kernels: Vec<Kernel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roofline_predictions: Vec<RooflinePrediction>,

    /// Forward-compatible capture of unknown top-level keys (D6 / `additionalProperties`).
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl ClsArtifact {
    /// Build an artifact stamped with the current [`SCHEMA_VERSION`], empty record arrays,
    /// and no unknown keys.
    pub fn new(session: Session) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            session,
            graph_breaks: Vec::new(),
            recompilations: Vec::new(),
            compiled_graphs: Vec::new(),
            compile_phases: Vec::new(),
            iterations: Vec::new(),
            lint_findings: Vec::new(),
            kernels: Vec::new(),
            roofline_predictions: Vec::new(),
            extra: JsonMap::new(),
        }
    }
}
