//! `cls-errors` — central error type for the compile-lens toolkit.
//!
//! [`ClsError`] is a typed enum with stable `CLS-EXXXX` codes per design discipline **D8**.
//! Every variant carries a code and an actionable hint; variants that wrap an underlying
//! failure also preserve a cause chain via `#[source]`. Rendering uses the miette
//! diagnostic protocol; see **ADR-022** for the *thiserror + miette* decision.
//!
//! # Doc links (deferred)
//!
//! The "doc link" part of D8 is intentionally **not committed** here: there is no public
//! docs site yet, and we do not want to bake a specific URL or domain into every variant.
//! The canonical in-repo doc page for each top-five code lives at
//! `docs/08_errors/<code>.md`. When a doc site exists, adding `url(...)` back to each
//! variant (or via a single helper) is a one-line change per variant — no
//! application-code impact.

// miette 7.x's `Diagnostic` derive expands code that trips `unused_assignments` on each
// variant's bound fields; the lint message itself suggests this allow. Scoped to this
// crate to keep real assignment bugs elsewhere visible.
#![allow(unused_assignments)]

use miette::Diagnostic;
use thiserror::Error;

/// Project-wide error type. Every user-facing error originates as one of these variants.
///
/// Codes are assigned monotonically (`CLS-E0001`..`CLS-E0010` for the initial set) and
/// are part of the public contract — they must not be renumbered or reused. New variants
/// take the next free `CLS-EXXXX`.
#[derive(Debug, Error, Diagnostic)]
pub enum ClsError {
    /// `CLS-E0001` — a file read attempt failed (missing path, permission denied, etc.).
    #[error("failed to read file: {path}")]
    #[diagnostic(
        code("CLS-E0001"),
        help("check that the path exists and the process has permission to read it")
    )]
    IoError {
        /// Path the read was attempted on.
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// `CLS-E0002` — `.cls.json` failed to parse as valid JSON / valid schema.
    #[error("failed to parse .cls.json artifact")]
    #[diagnostic(
        code("CLS-E0002"),
        help("the file must be valid JSON conforming to schema/v0.5.0.json")
    )]
    SchemaParseError {
        #[source]
        source: serde_json::Error,
    },

    /// `CLS-E0003` — the artifact's `schema_version` is not the one this analyzer expects.
    #[error("schema version mismatch: artifact is {found}, analyzer expects {expected}")]
    #[diagnostic(
        code("CLS-E0003"),
        help("run `cl migrate` if the version is within the N-1 compatibility window")
    )]
    SchemaVersionMismatch { found: String, expected: String },

    /// `CLS-E0004` — CLI argument validation failed.
    #[error("invalid CLI arguments: {detail}")]
    #[diagnostic(
        code("CLS-E0004"),
        help("see `cl --help` for usage; check spelling and required values")
    )]
    InvalidCliArgs { detail: String },

    /// `CLS-E0005` — a Python collector raised; the wrapped cause carries the detail.
    #[error("collector `{collector}` failed during capture")]
    #[diagnostic(
        code("CLS-E0005"),
        help("inspect the underlying cause; rerun with `CLS_LOG=debug` for more context")
    )]
    CollectorFailure {
        collector: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// `CLS-E0006` — an internal bug in an analyzer; please report.
    #[error("analyzer `{analyzer}` hit an internal error: {detail}")]
    #[diagnostic(
        code("CLS-E0006"),
        help("this is a bug in compile-lens; please file a report with the artifact attached")
    )]
    AnalyzerInternalError { analyzer: String, detail: String },

    /// `CLS-E0007` — the artifact must be migrated to the current schema first.
    #[error("schema migration required: from {from} to {to}")]
    #[diagnostic(
        code("CLS-E0007"),
        help("run `cl migrate <file>` to upgrade the artifact to the current schema")
    )]
    MigrationRequired { from: String, to: String },

    /// `CLS-E0008` — refused to record a path under `~/.ssh`/`~/.aws`/`~/.gnupg`
    /// (per `redaction_policy.md` §2.1).
    #[error("refused to record sensitive path: {path}")]
    #[diagnostic(
        code("CLS-E0008"),
        help("paths under ~/.ssh, ~/.aws, ~/.gnupg are never recorded (redaction_policy §2.1)")
    )]
    SensitivePathDetected { path: String },

    /// `CLS-E0009` — `cl scrub` refused to downgrade a redaction policy.
    /// Redaction is a one-way transform; re-collect with `--redaction <to>` to recover raw data.
    #[error("refused to demote redaction policy: {from} -> {to}")]
    #[diagnostic(
        code("CLS-E0009"),
        help("redaction is one-way; re-collect with `--redaction <level>` if raw data is needed")
    )]
    RedactionPolicyDemotionRefused { from: String, to: String },

    /// `CLS-E0010` — the MCP server refused a path outside its working-directory allowlist
    /// (Phase 2, see `threat_model.md` T5).
    #[error("path outside MCP working-directory allowlist: {requested}")]
    #[diagnostic(
        code("CLS-E0010"),
        help("the MCP sandbox only permits paths under the configured allowlist")
    )]
    WorkingDirectoryAllowlistViolation { requested: String },
}

impl ClsError {
    /// Return the stable `CLS-EXXXX` code as a `&'static str`.
    ///
    /// This is the inherent counterpart to [`miette::Diagnostic::code`]; the two are
    /// kept in sync by a registry test, so a divergence between the inherent match arms
    /// and the `#[diagnostic(code(...))]` attributes fails CI.
    pub fn code(&self) -> &'static str {
        match self {
            Self::IoError { .. } => "CLS-E0001",
            Self::SchemaParseError { .. } => "CLS-E0002",
            Self::SchemaVersionMismatch { .. } => "CLS-E0003",
            Self::InvalidCliArgs { .. } => "CLS-E0004",
            Self::CollectorFailure { .. } => "CLS-E0005",
            Self::AnalyzerInternalError { .. } => "CLS-E0006",
            Self::MigrationRequired { .. } => "CLS-E0007",
            Self::SensitivePathDetected { .. } => "CLS-E0008",
            Self::RedactionPolicyDemotionRefused { .. } => "CLS-E0009",
            Self::WorkingDirectoryAllowlistViolation { .. } => "CLS-E0010",
        }
    }
}
