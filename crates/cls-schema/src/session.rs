//! `session` object: run-level metadata. Semantic-domain hybrid layout (ADR-021) — atomic
//! scalars stay flat, structured fields ([`CompileConfig`], [`EnvSnapshot`]) nest.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{JsonMap, StringMap};

/// Redaction level recorded in the artifact; controls which fields the collector keeps at
/// capture time.
///
/// Modeled as a *closed* enum rather than `String` because it is the one session value
/// that drives security behavior, and it must **fail closed**: an unrecognized policy is a
/// hard deserialize error, never a silent fallback to a more permissive default. (Every
/// other constrained-string field in this crate stays `String` to honor D6 forward
/// compatibility — see the design note.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RedactionPolicy {
    /// Default: scrub host/command/paths and opt-in source excerpts.
    DefaultStrict,
    /// Internal sharing: relaxed scrubbing.
    Internal,
    /// Confidential: strictest classification.
    Confidential,
    /// Public-safe: cleared for public sharing.
    PublicSafe,
}

/// Run-level metadata (`session` object).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    // ===== required atomic scalars =====
    /// UUID v4.
    pub id: String,
    /// ISO-8601 / RFC-3339 timestamp.
    pub timestamp: String,
    pub torch_version: String,
    /// Redaction classification (D11). Required so an artifact can never be ambiguous
    /// about how it was scrubbed.
    pub redaction_policy: RedactionPolicy,

    // ===== optional atomic scalars (absent when unknown) =====
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triton_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_name: Option<String>,
    /// Redacted by default per `redaction_policy` (field-level redaction map, not a
    /// structural group — ADR-021).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    /// Redacted by default per `redaction_policy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration_count: Option<u64>,
    /// Reproducibility field (D10); kept flat because it is also a high-frequency
    /// standalone key (diff two sessions by commit).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_fingerprint: Option<String>,
    /// Reproducibility field (D10); flat because filtering distributed logs by rank is
    /// common.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_size: Option<u64>,

    // ===== nested structured fields =====
    /// Map of collector name -> version.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub collector_versions: StringMap,
    /// Mirrors the `torch.compile()` call (semantic domain: compile invocation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_config: Option<CompileConfig>,
    /// Runtime environment snapshot (semantic domain: `os.environ` + seed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_snapshot: Option<EnvSnapshot>,

    /// Forward-compatible capture of unknown session keys (D6 / `additionalProperties`).
    /// Session metadata is expected to grow (`gpu_arch`, `driver_version`, …).
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// The `torch.compile()` call parameters (reproducibility, D10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompileConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fullgraph: Option<bool>,
    /// `torch.compile(dynamic=...)`. Schema type is `[boolean, null]`: `null` means
    /// "auto" and is a distinct, meaningful state, so this field is **always serialized**
    /// (never skipped) to preserve a present-`null` across round trips.
    #[serde(default)]
    pub dynamic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Open map of arbitrary `torch.compile` options; cannot be flattened to named fields
    /// (ADR-021).
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub options: JsonMap,
}

/// Runtime environment captured at compile time (reproducibility, D10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvSnapshot {
    /// Whitelisted env vars known to affect `torch.compile`. Open map; values are
    /// redactable.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub relevant_env_vars: StringMap,
    /// Schema type is `[integer, null]`: a present-`null` seed (seed deliberately unset)
    /// differs from "no seed field at all", so this field is **always serialized**.
    #[serde(default)]
    pub random_seed: Option<u64>,
}
