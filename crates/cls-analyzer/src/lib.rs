//! `cls-analyzer` — Rust side of the toolkit's analysis layer.
//!
//! Each tool's analyzer lives in its own module. The shared shape is a
//! single `analyze` entry point that takes a parsed [`cls_schema::Session`]
//! and returns a tool-specific findings struct (`Result<_, ClsError>`).
//!
//! In Phase 1 the only wired analyzer is [`recompile`] (Tool 1 — recompile
//! aggregator). Its surface is in place — types declared, public functions
//! callable, the standard `#[tracing::instrument]` wrap on `analyze` — but
//! the implementation returns [`ClsError::NotYetImplemented`] until the
//! upcoming Phase 1 PRs land the clustering and suggestion logic.

pub mod recompile;
