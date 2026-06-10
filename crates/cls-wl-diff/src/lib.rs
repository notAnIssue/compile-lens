//! `cls-wl-diff` — the WL-signature graph diff for Tool 2a (compile-diff).
//!
//! The remaining pipeline (residual classification, `diff_graphs`) lands in later changes. So far
//! this crate provides the commutativity policy, the WL-signature computation and anchor
//! extraction the matcher seeds from, and the neighborhood expansion that grows those anchors into
//! a full matching.

mod commutativity;
mod diff;
mod expand;
mod render;
mod residual;
mod signature;

pub use commutativity::CommutativitySet;
pub use diff::{diff_graphs, IrGraphDiff, NodeId};
pub use expand::{anchor_uniqueness_ratio, expand_from_anchors, Matching};
pub use render::{render, Format};
pub use residual::{classify_residual, ResidualClassification};
pub use signature::{extract_anchors, Anchor, FxGraph};
