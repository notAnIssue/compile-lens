//! Tool 6 — CODA-style algebraic fusion-opportunity detector (crown-jewel feature).
//!
//! Finds `GEMM-Residual-RMSNorm-GEMM` (Pattern A) fusion opportunities in a `torch.compile` FX
//! graph that Inductor leaves on the table — cross-module algebraic fusions its local schedule
//! rules don't perform (the `1/rms` row-constant scale can be pulled through the second GEMM and
//! applied in its epilogue, removing the full-tensor RMSNorm materialization between the two GEMMs).
//! It is **suggest-only**: it reads the serialized graph, never mutates it, and never runs a kernel.
//!
//! Three disciplines bound the tool (design source-of-truth: `the design notes`):
//!   * **forward-only (N5)** — backward subgraphs are never analyzed;
//!   * **torch-concrete (N9/N12)** — patterns are named matchers over concrete `aten` ops, not a
//!     generic rewrite DSL; extending the tool means adding an `op_type` case, not a grammar;
//!   * **analytical roofline only (N10)** — the cost model is an HBM-bytes roofline estimate
//!     (reusing `cls-roofline`'s bandwidth registry), never a measured runtime.
//!
//! Per ADR-037 this is a Rust analyzer over the serialized `FxNode[]` (ADR-024), like `lint` and
//! `wl-diff` — no live Python graph, no re-trace. **This module is the scaffold**; the Pattern A
//! matcher, the cost model, and the renderer land in the following changes.

use cls_schema::{ClsArtifact, FusionOpportunity};

/// Scan a session's serialized FX graph for fusion opportunities.
///
/// Scaffold: returns nothing until the Pattern A matcher lands. Forward-only and suggest-only by
/// construction — it will read the serialized `FxNode[]` and never execute anything.
pub fn analyze(_artifact: &ClsArtifact) -> Vec<FusionOpportunity> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_artifact() -> ClsArtifact {
        let json = r#"{"schema_version":"0.5.0",
            "session":{"id":"00000000-0000-4000-8000-000000000000",
                       "timestamp":"2026-01-01T00:00:00Z",
                       "torch_version":"2.6.0","redaction_policy":"default-strict"}}"#;
        serde_json::from_str(json).expect("artifact parses")
    }

    #[test]
    fn scaffold_reports_no_opportunities_yet() {
        assert!(analyze(&empty_artifact()).is_empty());
    }
}
