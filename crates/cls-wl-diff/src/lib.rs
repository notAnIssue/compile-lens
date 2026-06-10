//! `cls-wl-diff` — the WL-signature graph diff for Tool 2a (compile-diff).
//!
//! The full matching pipeline (neighborhood expansion, residual classification, `diff_graphs`)
//! lands in later changes. So far this crate provides the commutativity policy and, on top of it,
//! the WL-signature computation and anchor extraction the matcher seeds from.

mod commutativity;
mod signature;

pub use commutativity::CommutativitySet;
pub use signature::{extract_anchors, Anchor, FxGraph};
