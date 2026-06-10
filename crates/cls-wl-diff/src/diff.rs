//! `diff_graphs` — the three-phase compile-diff, assembled (Tool 2a, design §8.2.5).
//!
//! This ties the phases into one pure function: WL-signature anchoring → neighborhood expansion →
//! residual classification. Given two captured graphs it returns an [`IrGraphDiff`] — what was
//! added, removed, modified, the matched pairs with their confidence, and two quality numbers
//! (match coverage and anchor uniqueness). No global state and no torch dependency: the same two
//! graphs always produce the same diff (D4), so it is fully unit-testable.

use crate::expand::expand_from_anchors;
use crate::residual::classify_residual;
use crate::signature::{extract_anchors, FxGraph};
use crate::CommutativitySet;

/// A node id within a graph.
pub type NodeId = String;

/// The structured diff between two compiled graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct IrGraphDiff {
    /// Head node ids with no base counterpart.
    pub added: Vec<NodeId>,
    /// Base node ids with no head counterpart.
    pub removed: Vec<NodeId>,
    /// Matched `(base, head)` pairs whose content changed (different attributes, or — recovered in
    /// residual — a non-commutative operand reorder).
    pub modified: Vec<(NodeId, NodeId)>,
    /// Every matched pair with its `[0, 1]` confidence: `(base, head, confidence)`.
    pub matched: Vec<(NodeId, NodeId, f64)>,
    /// Fraction of base nodes that found a match.
    pub match_coverage: f64,
    /// Fraction of the base graph's signatures owned by exactly one node (ambiguity score).
    pub anchor_uniqueness_ratio: f64,
}

/// Diff two compiled graphs: anchor by signature, expand the neighborhood, classify the residual.
pub fn diff_graphs(before: &FxGraph, after: &FxGraph) -> IrGraphDiff {
    let commutativity = CommutativitySet::standard();
    let anchors = extract_anchors(before, after, &commutativity);
    let matching = expand_from_anchors(before, after, &anchors, &commutativity);
    let residual = classify_residual(before, after, &matching, &commutativity);

    let matched = residual
        .matched
        .iter()
        .map(|(base, head)| {
            let confidence = residual.confidence.get(base).copied().unwrap_or(0.0);
            (base.clone(), head.clone(), confidence)
        })
        .collect();

    IrGraphDiff {
        added: residual.added,
        removed: residual.removed,
        modified: residual.modified,
        matched,
        match_coverage: residual.match_coverage,
        anchor_uniqueness_ratio: matching.anchor_uniqueness_ratio,
    }
}
