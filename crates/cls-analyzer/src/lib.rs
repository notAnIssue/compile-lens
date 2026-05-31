//! `cls-analyzer` — skeleton crate. Real implementation lands in a later change.
//!
//! Carries a sample `#[tracing::instrument]` so the workspace exercises the attribute
//! path (span entry/exit are emitted at INFO by default) — once a real analyzer function
//! lands, the same pattern applies without re-discovering the crate / feature setup.

/// Placeholder so the crate compiles as part of the workspace.
///
/// Wrapped in `#[tracing::instrument]` to verify the attribute compiles against the
/// workspace-pinned `tracing` and to seed the pattern for the real analyzer.
#[tracing::instrument]
pub fn placeholder() {
    tracing::debug!("cls-analyzer placeholder called");
}
