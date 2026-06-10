//! `cls-wl-diff` — the WL-signature graph diff for Tool 2a (compile-diff).
//!
//! The matching algorithm itself (signature computation, neighborhood expansion, residual
//! classification) lands in later changes. This change adds the first piece it depends on: the
//! commutativity policy that decides whether reordering a node's operands is a real change.

mod commutativity;

pub use commutativity::CommutativitySet;
