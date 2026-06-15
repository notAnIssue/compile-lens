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
    /// Matched `(base_id, head_id)` pairs whose content changed — different attributes, or a
    /// recovered non-commutative operand reorder. In matching order.
    pub modified: Vec<(String, String)>,
    /// All matched `(base_id, head_id)` pairs: the structural matches plus any recovered
    /// reordered pairs. The added/removed lists are exactly the nodes not in here.
    pub matched: Vec<(String, String)>,
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
    // Start from the structural matches, then recover reordered pairs: a base and head node left
    // unmatched but with the same op_type and the same *set* of matched inputs is the same node
    // with its operands rearranged (the non-commutative operand swap the signature flagged but
    // could not align by position). These are matches — and modifications.
    let mut all_matched: IndexMap<String, String> = matching.pairs.clone();
    let recovered = recover_reordered(before, after, &matching.pairs);
    for (base_id, head_id) in &recovered {
        all_matched.insert(base_id.clone(), head_id.clone());
    }

    let matched_base: IndexSet<&str> = all_matched.keys().map(String::as_str).collect();
    let matched_head: IndexSet<&str> = all_matched.values().map(String::as_str).collect();

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

    // A matched pair is `modified` when its content changed: a structural match whose attributes
    // differ, a recovered pair (operands reordered, matched only by `recover_reordered`), or a
    // *structurally matched* non-commutative pair whose operands were reordered — caught by
    // expansion via a stationary operand, so it never reached recovery (see `is_operand_reorder`).
    let recovered_set: IndexSet<&str> = recovered.iter().map(|(b, _)| b.as_str()).collect();
    let modified: Vec<(String, String)> = all_matched
        .iter()
        .filter(|(base_id, head_id)| {
            recovered_set.contains(base_id.as_str())
                || before.attrs_of(base_id) != after.attrs_of(head_id)
                || is_operand_reorder(before, after, &all_matched, commutativity, base_id, head_id)
        })
        .map(|(base_id, head_id)| (base_id.clone(), head_id.clone()))
        .collect();

    let matched: Vec<(String, String)> = all_matched
        .iter()
        .map(|(b, h)| (b.clone(), h.clone()))
        .collect();

    let match_coverage = if before.len() == 0 {
        1.0
    } else {
        all_matched.len() as f64 / before.len() as f64
    };

    // Structural-uniqueness component: 1 / (number of base nodes sharing this node's signature).
    let base_signatures = before.signatures(commutativity);
    let mut bucket_sizes: IndexMap<u64, usize> = IndexMap::new();
    for &sig in base_signatures.values() {
        *bucket_sizes.entry(sig).or_insert(0) += 1;
    }

    let confidence: IndexMap<String, f64> = all_matched
        .iter()
        .map(|(base_id, head_id)| {
            let uniqueness = base_signatures
                .get(base_id)
                .and_then(|sig| bucket_sizes.get(sig))
                .map_or(1.0, |&count| 1.0 / count as f64);
            let agreement = neighborhood_agreement(before, after, &all_matched, base_id, head_id);
            (base_id.clone(), 0.5 * uniqueness + 0.5 * agreement)
        })
        .collect();

    ResidualClassification {
        added,
        removed,
        modified,
        matched,
        match_coverage,
        confidence,
    }
}

/// Recover same-node pairs the structural matching left behind: a base and head node both
/// unmatched, with the same `op_type` and the same *multiset* of matched inputs (each base input
/// maps, through the structural matching, to the corresponding head input — regardless of order).
/// That is the same node with its operands rearranged, which the position-aligned expansion can't
/// catch precisely because the order changed. Greedy and 1:1; only nodes with a non-empty, fully
/// matched input list are eligible (so e.g. bare placeholders are never spuriously paired here).
fn recover_reordered(
    before: &FxGraph,
    after: &FxGraph,
    structural: &IndexMap<String, String>,
) -> Vec<(String, String)> {
    let matched_base: IndexSet<&str> = structural.keys().map(String::as_str).collect();
    let matched_head: IndexSet<&str> = structural.values().map(String::as_str).collect();

    let unmatched_head: Vec<&str> = after
        .node_ids()
        .filter(|id| !matched_head.contains(id))
        .collect();

    let mut used_head: IndexSet<&str> = IndexSet::new();
    let mut recovered: Vec<(String, String)> = Vec::new();

    for base in before.node_ids().filter(|id| !matched_base.contains(id)) {
        // Map base's in-graph inputs through the structural matching; require all matched.
        let mapped: Option<Vec<&str>> = before
            .inputs_of(base)
            .iter()
            .filter(|i| before.op_type_of(i).is_some())
            .map(|i| structural.get(i).map(String::as_str))
            .collect();
        let Some(mut mapped) = mapped else { continue };
        if mapped.is_empty() {
            continue;
        }
        mapped.sort_unstable();

        let base_op = before.op_type_of(base);
        for &head in &unmatched_head {
            if used_head.contains(head) || after.op_type_of(head) != base_op {
                continue;
            }
            let mut head_inputs: Vec<&str> = after
                .inputs_of(head)
                .iter()
                .filter(|i| after.op_type_of(i).is_some())
                .map(String::as_str)
                .collect();
            head_inputs.sort_unstable();
            if head_inputs == mapped {
                recovered.push((base.to_string(), head.to_string()));
                used_head.insert(head);
                break;
            }
        }
    }
    recovered
}

/// True when a *matched* non-commutative pair has the same operands in a *different* order — an
/// operand reorder neighborhood expansion matched (via a stationary operand that kept its position)
/// but never flagged. This is the partial-reorder counterpart to [`recover_reordered`]: that one
/// handles pairs expansion left *unmatched* (a full swap, where no operand keeps its position); when
/// at least one operand stays put the pair gets matched, so the order change has to be caught here
/// at classification time. `aten.addmm(bias, mat1, mat2)` → `aten.addmm(bias, mat2, mat1)` (every
/// biased `nn.Linear`) is the canonical case.
///
/// Commutative ops are exempt: their operand order is not semantic, and the signature already sorts
/// their inputs (so a commutative reorder matches cleanly and must stay silent). Only fires when the
/// in-graph operands are the same multiset in a different order — a genuine reorder, never a
/// different-operand change (those surface structurally elsewhere).
fn is_operand_reorder(
    before: &FxGraph,
    after: &FxGraph,
    matched: &IndexMap<String, String>,
    commutativity: &CommutativitySet,
    base: &str,
    head: &str,
) -> bool {
    let Some(op) = before.op_type_of(base) else {
        return false;
    };
    if commutativity.is_commutative(op) {
        return false;
    }
    // Base's in-graph inputs, in order, mapped through the matching; require all matched so the
    // comparison is well-defined (an unmatched input means we can't tell — stay silent).
    let mapped: Option<Vec<&str>> = before
        .inputs_of(base)
        .iter()
        .filter(|i| before.op_type_of(i).is_some())
        .map(|i| matched.get(i).map(String::as_str))
        .collect();
    let Some(mapped) = mapped else { return false };
    let head_seq: Vec<&str> = after
        .inputs_of(head)
        .iter()
        .filter(|i| after.op_type_of(i).is_some())
        .map(String::as_str)
        .collect();
    if mapped.len() < 2 || mapped.len() != head_seq.len() {
        return false; // need ≥2 in-graph operands of equal count to have a reorder
    }
    // Same operands (as a multiset) but a different order == a reorder.
    let mut mapped_sorted = mapped.clone();
    mapped_sorted.sort_unstable();
    let mut head_sorted = head_seq.clone();
    head_sorted.sort_unstable();
    mapped_sorted == head_sorted && mapped != head_seq
}

/// Fraction of `base`'s in-graph neighbors (inputs + consumers) whose match is a neighbor of
/// `head`. `1.0` when `base` has no in-graph neighbors (nothing to disagree about).
fn neighborhood_agreement(
    before: &FxGraph,
    after: &FxGraph,
    pairs: &IndexMap<String, String>,
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
            pairs
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
            out_shape: Vec::new(),
            out_dtype: None,
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
