//! Neighborhood expansion for the compile-diff (Tool 2a, design §8.2.5 Phase 2).
//!
//! Anchors (from the signature phase) are the nodes we are sure about. Most nodes are not unique
//! enough to anchor on their own, but they sit next to ones that are. This phase grows the match
//! outward from each anchor by breadth-first search over both graphs at once, up to a small radius
//! `d_max = 3`: a node next to an already-matched pair is matched to the corresponding node on the
//! other side when they line up — same operand position along an edge and same `op_type`.
//!
//! The matching is **greedy**: the first valid counterpart wins, and a node is matched at most once
//! on each side (it never re-pairs). Two edge directions are followed from a matched pair `(b, h)`:
//!
//! - **Inputs by position.** The k-th input of `b` is matched to the k-th input of `h` — operand
//!   position is the alignment, which is why a non-commutative operand swap (caught already in the
//!   signature) also keeps inputs from lining up here.
//! - **Consumers by position.** A consumer of `b` is matched to a consumer of `h` when `b` and `h`
//!   feed them at the same operand position.
//!
//! Alongside the matching we report an **ambiguity score**, `anchor_uniqueness_ratio`: the fraction
//! of a graph's distinct signatures that belong to exactly one node. Near 1.0 means almost every
//! node is structurally unique (anchoring is reliable); lower means many look-alike nodes (matches
//! are less certain), which downstream confidence reporting can surface.

use std::collections::VecDeque;

use indexmap::{IndexMap, IndexSet};

use crate::signature::{Anchor, FxGraph};
use crate::CommutativitySet;

/// How far from an anchor the expansion reaches. A hyperparameter, not a correctness bound
/// (design §8.2.5): three hops covers a node's local structure without letting one anchor's match
/// run away across the whole graph.
const D_MAX: usize = 3;

/// The result of expanding anchors over a pair of graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Matching {
    /// base node id → head node id, including the seed anchors and every expanded match, in the
    /// order they were established (anchors first). Deterministic.
    pub pairs: IndexMap<String, String>,
    /// Ambiguity score of the base graph in `[0, 1]`: fraction of signatures owned by exactly one
    /// node. Higher is less ambiguous.
    pub anchor_uniqueness_ratio: f64,
}

/// Grow `anchors` into a full matching by neighborhood expansion (see module docs).
pub fn expand_from_anchors(
    before: &FxGraph,
    after: &FxGraph,
    anchors: &[Anchor],
    commutativity: &CommutativitySet,
) -> Matching {
    let mut pairs: IndexMap<String, String> = IndexMap::new();
    let mut matched_head: IndexSet<String> = IndexSet::new();
    let mut queue: VecDeque<(String, String, usize)> = VecDeque::new();

    for anchor in anchors {
        if pairs.contains_key(&anchor.base_id) || matched_head.contains(&anchor.head_id) {
            continue; // defensive: ignore a duplicated/contradictory anchor
        }
        pairs.insert(anchor.base_id.clone(), anchor.head_id.clone());
        matched_head.insert(anchor.head_id.clone());
        queue.push_back((anchor.base_id.clone(), anchor.head_id.clone(), 0));
    }

    while let Some((base, head, depth)) = queue.pop_front() {
        if depth >= D_MAX {
            continue; // matched, but too deep to expand further
        }

        // ── inputs, aligned by operand position ──────────────────────────────────────────
        let base_inputs = before.inputs_of(&base).to_vec();
        let head_inputs = after.inputs_of(&head).to_vec();
        for k in 0..base_inputs.len().min(head_inputs.len()) {
            try_match(
                &base_inputs[k],
                &head_inputs[k],
                depth + 1,
                before,
                after,
                &mut pairs,
                &mut matched_head,
                &mut queue,
            );
        }

        // ── consumers, aligned by the position at which (base, head) feed them ────────────
        for consumer_base in before.consumers_of(&base) {
            let pos_base = before.input_position(consumer_base, &base);
            for consumer_head in after.consumers_of(&head) {
                if after.input_position(consumer_head, &head) != pos_base {
                    continue;
                }
                // Greedy: the first position-aligned, type-matching, free counterpart wins.
                if try_match(
                    consumer_base,
                    consumer_head,
                    depth + 1,
                    before,
                    after,
                    &mut pairs,
                    &mut matched_head,
                    &mut queue,
                ) {
                    break;
                }
            }
        }
    }

    Matching {
        pairs,
        anchor_uniqueness_ratio: anchor_uniqueness_ratio(before, commutativity),
    }
}

/// Match `base_id` to `head_id` if both are real, unmatched, and share an `op_type`. Returns
/// whether a new match was made (and, if so, enqueues it for further expansion).
#[allow(clippy::too_many_arguments)]
fn try_match(
    base_id: &str,
    head_id: &str,
    depth: usize,
    before: &FxGraph,
    after: &FxGraph,
    pairs: &mut IndexMap<String, String>,
    matched_head: &mut IndexSet<String>,
    queue: &mut VecDeque<(String, String, usize)>,
) -> bool {
    if pairs.contains_key(base_id) || matched_head.contains(head_id) {
        return false;
    }
    let (Some(base_op), Some(head_op)) = (before.op_type_of(base_id), after.op_type_of(head_id))
    else {
        return false; // one of them isn't a node in its graph (e.g. an external input ref)
    };
    if base_op != head_op {
        return false;
    }
    pairs.insert(base_id.to_string(), head_id.to_string());
    matched_head.insert(head_id.to_string());
    queue.push_back((base_id.to_string(), head_id.to_string(), depth));
    true
}

/// Fraction of a graph's distinct signatures that belong to exactly one node — the ambiguity score
/// (design §8.2.5). `1.0` when every node is structurally unique; lower with more look-alikes. An
/// empty graph is vacuously `1.0`.
pub fn anchor_uniqueness_ratio(graph: &FxGraph, commutativity: &CommutativitySet) -> f64 {
    let signatures = graph.signatures(commutativity);
    if signatures.is_empty() {
        return 1.0;
    }
    let mut bucket_sizes: IndexMap<u64, usize> = IndexMap::new();
    for &sig in signatures.values() {
        *bucket_sizes.entry(sig).or_insert(0) += 1;
    }
    let unique = bucket_sizes.values().filter(|&&count| count == 1).count();
    unique as f64 / bucket_sizes.len() as f64
}

#[cfg(test)]
mod tests {
    use cls_schema::FxNode;

    use super::*;
    use crate::signature::extract_anchors;

    fn node(id: &str, op_type: &str, inputs: &[&str]) -> FxNode {
        FxNode {
            id: id.to_string(),
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            attrs: Default::default(),
        }
    }

    fn matching(base: &[FxNode], head: &[FxNode]) -> Matching {
        let before = FxGraph::from_nodes(base);
        let after = FxGraph::from_nodes(head);
        let comm = CommutativitySet::standard();
        let anchors = extract_anchors(&before, &after, &comm);
        expand_from_anchors(&before, &after, &anchors, &comm)
    }

    /// Ambiguous neighbors one hop from an anchor are matched by expansion. `p` anchors (the only
    /// placeholder, identical consumer set on both sides); its two relu consumers share a signature
    /// so neither anchors on its own — only expansion can pair them.
    #[test]
    fn test_expansion_d1() {
        let base = [
            node("p", "placeholder", &[]),
            node("r1", "aten.relu.default", &["p"]),
            node("r2", "aten.relu.default", &["p"]),
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("s1", "aten.relu.default", &["q"]),
            node("s2", "aten.relu.default", &["q"]),
        ];
        let result = matching(&base, &head);
        assert!(result.pairs.contains_key("r1"));
        assert!(result.pairs.contains_key("r2"));
    }

    /// Expansion reaches exactly `d_max = 3` hops and no further. Two mirror relu chains hang off
    /// the sole anchor `a`; the chains are pairwise ambiguous (so only expansion matches them), and
    /// the node four hops out is left unmatched.
    #[test]
    fn test_expansion_d3_max() {
        let base = [
            node("a", "placeholder", &[]),
            node("x0", "aten.relu.default", &["a"]),
            node("x1", "aten.relu.default", &["x0"]),
            node("x2", "aten.relu.default", &["x1"]),
            node("x3", "aten.relu.default", &["x2"]),
            node("y0", "aten.relu.default", &["a"]),
            node("y1", "aten.relu.default", &["y0"]),
            node("y2", "aten.relu.default", &["y1"]),
            node("y3", "aten.relu.default", &["y2"]),
        ];
        let head = [
            node("b", "placeholder", &[]),
            node("u0", "aten.relu.default", &["b"]),
            node("u1", "aten.relu.default", &["u0"]),
            node("u2", "aten.relu.default", &["u1"]),
            node("u3", "aten.relu.default", &["u2"]),
            node("v0", "aten.relu.default", &["b"]),
            node("v1", "aten.relu.default", &["v0"]),
            node("v2", "aten.relu.default", &["v1"]),
            node("v3", "aten.relu.default", &["v2"]),
        ];
        let result = matching(&base, &head);
        // a anchors; x2 is three hops from it (matched), x3 is four (beyond d_max, unmatched).
        assert!(
            result.pairs.contains_key("x2"),
            "x2 (distance 3) should match"
        );
        assert!(
            !result.pairs.contains_key("x3"),
            "x3 (distance 4) is beyond d_max"
        );
    }

    /// When a matched pair has several position-aligned consumer candidates, the first valid one
    /// wins (greedy, deterministic). `p` anchors; its two ambiguous relu consumers are paired in
    /// order, so the first base consumer takes the first head consumer.
    #[test]
    fn test_expansion_greedy_first_wins() {
        let base = [
            node("p", "placeholder", &[]),
            node("c1", "aten.relu.default", &["p"]),
            node("c2", "aten.relu.default", &["p"]),
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("d1", "aten.relu.default", &["q"]),
            node("d2", "aten.relu.default", &["q"]),
        ];
        let result = matching(&base, &head);
        assert_eq!(result.pairs.get("c1").map(String::as_str), Some("d1"));
        assert_eq!(result.pairs.get("c2").map(String::as_str), Some("d2"));
    }

    /// A graph whose nodes all have distinct signatures scores near 1.0.
    #[test]
    fn test_anchor_uniqueness_high_when_distinct_signatures() {
        let nodes = [
            node("p", "placeholder", &[]),
            node("a", "aten.abs.default", &["p"]),
            node("b", "aten.neg.default", &["a"]),
        ];
        let graph = FxGraph::from_nodes(&nodes);
        let ratio = anchor_uniqueness_ratio(&graph, &CommutativitySet::standard());
        assert!((ratio - 1.0).abs() < 1e-9, "expected ~1.0, got {ratio}");
    }

    /// A graph with structurally identical nodes scores below 1.0.
    #[test]
    fn test_anchor_uniqueness_low_when_repeated_signatures() {
        // Two identical relu-of-input nodes collide; the placeholder is unique. 2 buckets, 1 unique.
        let nodes = [
            node("p", "placeholder", &[]),
            node("r1", "aten.relu.default", &["p"]),
            node("r2", "aten.relu.default", &["p"]),
        ];
        let graph = FxGraph::from_nodes(&nodes);
        let ratio = anchor_uniqueness_ratio(&graph, &CommutativitySet::standard());
        assert!(
            ratio < 1.0,
            "expected < 1.0 with repeated signatures, got {ratio}"
        );
        assert!(
            (ratio - 0.5).abs() < 1e-9,
            "expected 0.5 (1 unique of 2 buckets), got {ratio}"
        );
    }

    /// Nodes unreachable from any anchor are simply left unmatched — no panic, no spurious match.
    #[test]
    fn test_expansion_handles_disconnected_components() {
        // p->r is a connected, anchorable component. i1/i2 are an ambiguous island consuming an
        // out-of-graph value, unreachable from any anchor, so expansion never touches them.
        let base = [
            node("p", "placeholder", &[]),
            node("r", "aten.relu.default", &["p"]),
            node("i1", "aten.abs.default", &["external"]),
            node("i2", "aten.abs.default", &["external"]),
        ];
        let head = [
            node("q", "placeholder", &[]),
            node("s", "aten.relu.default", &["q"]),
            node("j1", "aten.abs.default", &["external"]),
            node("j2", "aten.abs.default", &["external"]),
        ];
        let result = matching(&base, &head);
        // The connected p->r component matches; the disconnected ambiguous island is not reached.
        assert!(result.pairs.contains_key("r"));
        assert!(!result.pairs.contains_key("i1"));
    }
}
