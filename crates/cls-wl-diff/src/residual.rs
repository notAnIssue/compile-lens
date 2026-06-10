//! Residual classification for the compile-diff (Tool 2a, design notes Phase 3).
//!
//! After anchoring and neighborhood expansion, some nodes are matched and some are not. This phase
//! reads off the actual diff from that matching:
//!
//! - **added** — a head node with no base counterpart (it appeared);
//! - **removed** — a base node with no head counterpart (it went away);
//! - **modified** — a *matched* pair whose attributes differ. The signature ignores attributes (it
//!   hashes structure, not constant values), so a node that only changed an attribute — say
//!   `add(a, b, alpha=1)` becoming `add(a, b, alpha=2)` — still matches structurally; comparing the
//!   attrs of the matched pair is what surfaces the change.
//!
//! It also reports two quality numbers: `match_coverage` (the fraction of base nodes that found a
//! match) and a per-match `confidence` in `[0, 1]`, blending how structurally unique the matched
//! node is with how much its neighborhood agrees (its matched neighbors map to the counterpart's
//! neighbors). The matching is approximate, so these let downstream weigh how much to trust it.

use indexmap::{IndexMap, IndexSet};

use crate::expand::Matching;
use crate::signature::FxGraph;
use crate::CommutativitySet;

/// The classified diff plus its quality numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualClassification {
    /// Head node ids with no base counterpart, in node order.
    pub added: Vec<String>,
    /// Base node ids with no head counterpart, in node order.
    pub removed: Vec<String>,
    /// Matched `(base_id, head_id)` pairs whose attributes differ, in matching order.
    pub modified: Vec<(String, String)>,
    /// Fraction of base nodes that found a match, in `[0, 1]`. An empty base graph is `1.0`.
    pub match_coverage: f64,
    /// Per matched base node, a confidence in `[0, 1]` (structural uniqueness blended with
    /// neighborhood agreement). Higher means a more trustworthy match.
    pub confidence: IndexMap<String, f64>,
}

/// Classify the residual of a matching into added / removed / modified, with coverage and
/// per-match confidence (see module docs).
pub fn classify_residual(
    before: &FxGraph,
    after: &FxGraph,
    matching: &Matching,
    commutativity: &CommutativitySet,
) -> ResidualClassification {
    let matched_base: IndexSet<&str> = matching.pairs.keys().map(String::as_str).collect();
    let matched_head: IndexSet<&str> = matching.pairs.values().map(String::as_str).collect();

    let removed: Vec<String> = before
        .node_ids()
        .filter(|id| !matched_base.contains(id))
        .map(String::from)
        .collect();
    let added: Vec<String> = after
        .node_ids()
        .filter(|id| !matched_head.contains(id))
        .map(String::from)
        .collect();

    // A matched pair is `modified` when its attributes differ — structure already matched, since
    // the signature does not look at attrs.
    let modified: Vec<(String, String)> = matching
        .pairs
        .iter()
        .filter(|(base_id, head_id)| before.attrs_of(base_id) != after.attrs_of(head_id))
        .map(|(base_id, head_id)| (base_id.clone(), head_id.clone()))
        .collect();

    let match_coverage = if before.len() == 0 {
        1.0
    } else {
        matching.pairs.len() as f64 / before.len() as f64
    };

    // Structural-uniqueness component: 1 / (number of base nodes sharing this node's signature).
    let base_signatures = before.signatures(commutativity);
    let mut bucket_sizes: IndexMap<u64, usize> = IndexMap::new();
    for &sig in base_signatures.values() {
        *bucket_sizes.entry(sig).or_insert(0) += 1;
    }

    let confidence: IndexMap<String, f64> = matching
        .pairs
        .iter()
        .map(|(base_id, head_id)| {
            let uniqueness = base_signatures
                .get(base_id)
                .and_then(|sig| bucket_sizes.get(sig))
                .map_or(1.0, |&count| 1.0 / count as f64);
            let agreement = neighborhood_agreement(before, after, matching, base_id, head_id);
            (base_id.clone(), 0.5 * uniqueness + 0.5 * agreement)
        })
        .collect();

    ResidualClassification {
        added,
        removed,
        modified,
        match_coverage,
        confidence,
    }
}

/// Fraction of `base`'s in-graph neighbors (inputs + consumers) whose match is a neighbor of
/// `head`. `1.0` when `base` has no in-graph neighbors (nothing to disagree about).
fn neighborhood_agreement(
    before: &FxGraph,
    after: &FxGraph,
    matching: &Matching,
    base: &str,
    head: &str,
) -> f64 {
    let base_neighbors: Vec<&str> = before
        .inputs_of(base)
        .iter()
        .map(String::as_str)
        .filter(|id| before.op_type_of(id).is_some()) // drop refs to out-of-graph values
        .chain(before.consumers_of(base))
        .collect();
    if base_neighbors.is_empty() {
        return 1.0;
    }

    let head_neighbors: IndexSet<&str> = after
        .inputs_of(head)
        .iter()
        .map(String::as_str)
        .filter(|id| after.op_type_of(id).is_some())
        .chain(after.consumers_of(head))
        .collect();

    let agreeing = base_neighbors
        .iter()
        .filter(|bn| {
            matching
                .pairs
                .get(**bn)
                .is_some_and(|hn| head_neighbors.contains(hn.as_str()))
        })
        .count();
    agreeing as f64 / base_neighbors.len() as f64
}

#[cfg(test)]
mod tests {
    use cls_schema::FxNode;

    use super::*;
    use crate::expand_from_anchors;
    use crate::signature::extract_anchors;

    fn node(id: &str, op_type: &str, inputs: &[&str]) -> FxNode {
        FxNode {
            id: id.to_string(),
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            attrs: Default::default(),
        }
    }

    fn node_attr(id: &str, op_type: &str, inputs: &[&str], key: &str, value: i64) -> FxNode {
        let mut n = node(id, op_type, inputs);
        n.attrs.insert(key.to_string(), serde_json::json!(value));
        n
    }

    fn classify(base: &[FxNode], head: &[FxNode]) -> ResidualClassification {
        let before = FxGraph::from_nodes(base);
        let after = FxGraph::from_nodes(head);
        let comm = CommutativitySet::standard();
        let anchors = extract_anchors(&before, &after, &comm);
        let matching = expand_from_anchors(&before, &after, &anchors, &comm);
        classify_residual(&before, &after, &matching, &comm)
    }

    /// A node present only in head is classified `added`, nothing else.
    #[test]
    fn test_residual_add_only() {
        let base = [
            node("p", "placeholder", &[]),
            node("r", "aten.relu.default", &["p"]),
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("s", "aten.relu.default", &["q"]),
            node("t", "aten.abs.default", &["s"]), // only in head
        ];
        let result = classify(&base, &head);
        assert_eq!(result.added, ["t"]);
        assert!(result.removed.is_empty());
        assert!(result.modified.is_empty());
    }

    /// A node present only in base is classified `removed`, nothing else.
    #[test]
    fn test_residual_remove_only() {
        let base = [
            node("p", "placeholder", &[]),
            node("r", "aten.relu.default", &["p"]),
            node("t", "aten.abs.default", &["r"]), // only in base
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("s", "aten.relu.default", &["q"]),
        ];
        let result = classify(&base, &head);
        assert_eq!(result.removed, ["t"]);
        assert!(result.added.is_empty());
        assert!(result.modified.is_empty());
    }

    /// A matched pair whose attribute differs is classified `modified` — structure matched, attr
    /// changed.
    #[test]
    fn test_residual_modify_attribute() {
        let base = [
            node("p", "placeholder", &[]),
            node_attr("n", "aten.relu.default", &["p"], "scale", 1),
        ];
        let head = [
            node("q", "placeholder", &[]),
            node_attr("m", "aten.relu.default", &["q"], "scale", 2),
        ];
        let result = classify(&base, &head);
        assert_eq!(result.modified, [("n".to_string(), "m".to_string())]);
        assert!(result.added.is_empty());
        assert!(result.removed.is_empty());
    }

    /// Coverage is matched-base over total-base. With one of three base nodes unmatched, 2/3.
    #[test]
    fn test_match_coverage_calculation() {
        let base = [
            node("p", "placeholder", &[]),
            node("r", "aten.relu.default", &["p"]),
            node("t", "aten.abs.default", &["r"]), // unmatched (removed)
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("s", "aten.relu.default", &["q"]),
        ];
        let result = classify(&base, &head);
        assert!(
            (result.match_coverage - 2.0 / 3.0).abs() < 1e-9,
            "expected 2/3, got {}",
            result.match_coverage
        );
    }

    /// Every per-match confidence lands in [0, 1], and matched pairs each get one.
    #[test]
    fn test_confidence_score_range_0_1() {
        let base = [
            node("p", "placeholder", &[]),
            node("a", "aten.abs.default", &["p"]),
            node("b", "aten.neg.default", &["a"]),
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("c", "aten.abs.default", &["q"]),
            node("d", "aten.neg.default", &["c"]),
        ];
        let result = classify(&base, &head);
        assert!(!result.confidence.is_empty());
        for (id, &score) in &result.confidence {
            assert!(
                (0.0..=1.0).contains(&score),
                "{id} confidence {score} out of [0,1]"
            );
        }
    }
}
