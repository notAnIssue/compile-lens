//! `cls-analyzer` — Rust side of the toolkit's analysis layer.
//!
//! Each tool's analyzer lives in its own module. The shared shape is a
//! single `analyze` entry point that takes a parsed [`cls_schema::ClsArtifact`]
//! and returns a tool-specific findings struct (`Result<_, ClsError>`).
//!
//! In Phase 1 the wired analyzer is [`recompile`] (Tool 1 — recompile aggregator):
//! [`recompile::analyze`] clusters a session's guard failures into dynamic-axis-attributed
//! findings ([`recompile_cluster`] holds the algorithm). Ranked suggestions land in a later PR.

pub mod cache_stability;
pub mod divergence;
pub mod lint;
pub mod recompile;
pub mod recompile_cluster;
pub mod recompile_diff;
pub mod recompile_render;
pub mod recompile_suggest;
