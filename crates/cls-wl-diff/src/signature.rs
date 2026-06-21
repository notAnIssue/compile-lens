//! WL-signature computation for the compile-diff (Tool 2a).
//!
//! To diff two compiled graphs we cannot match nodes by id — ids are unstable across compiles.
//! Instead each node gets a **structural signature**: a hash of what the node *is* and what it is
//! connected to, computed so that two nodes with the same local structure get the same signature
//! regardless of their names. Nodes whose signature is unique within their graph and equal across
//! the two graphs become **anchors** — the high-confidence seed matches a later neighborhood
//! expansion grows from.
//!
//! The signature of a node is a hash of three things:
//! 1. its `op_type`;
//! 2. its inputs' signatures — **sorted** for a commutative op (so reordering operands is
//!    invisible, `add(a, b)` == `add(b, a)`) but **kept in order** for a non-commutative op (so
//!    `sub(a, b)` differs from `sub(b, a)`); commutativity comes from [`CommutativitySet`];
//! 3. the sorted `op_type`s of its consumers (the nodes that read its output).
//!
//! Two deliberate properties of this definition:
//! - **It uses no node names.** Renaming every node leaves all signatures unchanged, so a pure
//!   rename is matched rather than mistaken for a rewrite.
//! - **Placeholders are seeded with their argument index.** Two bare graph inputs would otherwise
//!   be indistinguishable (same op, same consumers) and an operand swap between them would be
//!   invisible. Seeding by argument position distinguishes the first input from the second, which
//!   is genuine identity (and stays stable under renames and internal node add/remove, since the
//!   argument list does not shift). See the note's design section.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use cls_schema::FxNode;
use indexmap::IndexMap;

use crate::CommutativitySet;

/// The op_type marking a graph-input node, whose signature is seeded by argument index.
const PLACEHOLDER: &str = "placeholder";

/// An in-memory view of a captured FX graph, built from the `compiled_graphs[].nodes[]` contract.
///
/// Holds the nodes, an id→position index, the reverse edges (consumers) needed for the signature,
/// and each placeholder's argument index. Construction is `O(V + E)`.
pub struct FxGraph {
    nodes: Vec<FxNode>,
    index_of: IndexMap<String, usize>,
    /// For each node position, the positions of the nodes that consume its output.
    consumers: Vec<Vec<usize>>,
    /// id → argument index, for placeholder nodes only.
    placeholder_arg: IndexMap<String, usize>,
}

impl FxGraph {
    /// Build a graph from the contract's node list. Inputs that reference an unknown id (e.g. an
    /// external value not in this graph) are simply not wired as edges.
    pub fn from_nodes(nodes: &[FxNode]) -> Self {
        let mut index_of = IndexMap::with_capacity(nodes.len());
        let mut placeholder_arg = IndexMap::new();
        for (pos, node) in nodes.iter().enumerate() {
            index_of.insert(node.id.clone(), pos);
            if node.op_type == PLACEHOLDER {
                let arg = placeholder_arg.len();
                placeholder_arg.insert(node.id.clone(), arg);
            }
        }

        let mut consumers = vec![Vec::new(); nodes.len()];
        for (pos, node) in nodes.iter().enumerate() {
            for input_id in &node.inputs {
                if let Some(&producer) = index_of.get(input_id) {
                    consumers[producer].push(pos);
                }
            }
        }

        Self {
            nodes: nodes.to_vec(),
            index_of,
            consumers,
            placeholder_arg,
        }
    }

    /// op_type of a node by id, or `None` if the id is not in the graph.
    pub(crate) fn op_type_of(&self, id: &str) -> Option<&str> {
        self.index_of
            .get(id)
            .map(|&pos| self.nodes[pos].op_type.as_str())
    }

    /// The node's input ids in operand order (empty if the id is unknown). Some entries may
    /// reference values not present in this graph; callers filter as needed.
    pub(crate) fn inputs_of(&self, id: &str) -> &[String] {
        self.index_of
            .get(id)
            .map(|&pos| self.nodes[pos].inputs.as_slice())
            .unwrap_or(&[])
    }

    /// Ids of the nodes that consume this node's output, in node order.
    pub(crate) fn consumers_of(&self, id: &str) -> Vec<&str> {
        self.index_of
            .get(id)
            .map(|&pos| {
                self.consumers[pos]
                    .iter()
                    .map(|&c| self.nodes[c].id.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The operand position at which `producer` feeds `consumer`, or `None` if it does not (or
    /// either id is unknown). When a producer feeds a consumer more than once, the first position.
    pub(crate) fn input_position(&self, consumer: &str, producer: &str) -> Option<usize> {
        self.inputs_of(consumer).iter().position(|i| i == producer)
    }

    /// All node ids, in node order.
    pub(crate) fn node_ids(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|n| n.id.as_str())
    }

    /// Number of nodes in the graph.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// A node's attributes (the `attrs` map), or `None` if the id is unknown. Two structurally
    /// matched nodes whose attrs differ are what residual classification calls `modified`.
    pub(crate) fn attrs_of(&self, id: &str) -> Option<&cls_schema::JsonMap> {
        self.index_of.get(id).map(|&pos| &self.nodes[pos].attrs)
    }

    /// Compute every node's WL-signature, returned as id → signature in node order.
    ///
    /// Deterministic: same graph in, byte-identical signatures out (it uses a fixed-key hasher and
    /// sorts the parts that have no inherent order). Signatures are memoized and computed lazily
    /// over the DAG — a node's signature depends on its inputs', which (the FX graph being acyclic)
    /// always resolve without cycles.
    pub fn signatures(&self, commutativity: &CommutativitySet) -> IndexMap<String, u64> {
        let mut memo: Vec<Option<u64>> = vec![None; self.nodes.len()];
        let mut out = IndexMap::with_capacity(self.nodes.len());
        for pos in 0..self.nodes.len() {
            let sig = self.signature_at(pos, commutativity, &mut memo);
            out.insert(self.nodes[pos].id.clone(), sig);
        }
        out
    }

    fn signature_at(
        &self,
        pos: usize,
        commutativity: &CommutativitySet,
        memo: &mut Vec<Option<u64>>,
    ) -> u64 {
        if let Some(cached) = memo[pos] {
            return cached;
        }
        let node = &self.nodes[pos];

        let mut hasher = DefaultHasher::new();
        node.op_type.hash(&mut hasher);
        if let Some(arg) = self.placeholder_arg.get(&node.id) {
            // A graph input's identity is its argument position, full stop. Its signature is
            // seeded by that index (so the first input differs from the second) and deliberately
            // excludes inputs (it has none) and consumers — an input is the same input regardless
            // of what reads it, so it must still anchor when a downstream op is restructured (e.g.
            // a fusion split changes which ops consume it).
            arg.hash(&mut hasher);
        } else {
            // Inputs' signatures, in operand order; sorted only when the op is commutative.
            let mut input_sigs: Vec<u64> = node
                .inputs
                .iter()
                .filter_map(|id| self.index_of.get(id).copied())
                .map(|producer| self.signature_at(producer, commutativity, memo))
                .collect();
            if commutativity.is_commutative(&node.op_type) {
                input_sigs.sort_unstable();
            }
            // Consumers contribute only their op_types, sorted (their order is not meaningful).
            let mut consumer_ops: Vec<&str> = self.consumers[pos]
                .iter()
                .map(|&c| self.nodes[c].op_type.as_str())
                .collect();
            consumer_ops.sort_unstable();

            input_sigs.hash(&mut hasher);
            consumer_ops.hash(&mut hasher);
        }
        let sig = hasher.finish();

        memo[pos] = Some(sig);
        sig
    }
}

/// A high-confidence seed match: a node whose signature is unique within each graph and equal in
/// both, so the two are almost certainly the same node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Anchor {
    pub base_id: String,
    pub head_id: String,
    pub signature: u64,
}

/// Pair up the anchors between two graphs.
///
/// A signature anchors only if it occurs **exactly once** in the base graph and **exactly once** in
/// the head graph: a signature shared by several nodes is ambiguous and is left for neighborhood
/// expansion to resolve, not used as a seed. Result is sorted by base id for determinism.
pub fn extract_anchors(
    before: &FxGraph,
    after: &FxGraph,
    commutativity: &CommutativitySet,
) -> Vec<Anchor> {
    let base_unique = unique_signatures(&before.signatures(commutativity));
    let head_unique = unique_signatures(&after.signatures(commutativity));

    let mut anchors: Vec<Anchor> = base_unique
        .iter()
        .filter_map(|(sig, base_id)| {
            head_unique.get(sig).map(|head_id| Anchor {
                base_id: base_id.clone(),
                head_id: head_id.clone(),
                signature: *sig,
            })
        })
        .collect();
    anchors.sort_by(|a, b| a.base_id.cmp(&b.base_id));
    anchors
}

/// Map each signature that occurs exactly once to its node id; drop signatures shared by more than
/// one node (they are not unique, so not anchorable).
fn unique_signatures(signatures: &IndexMap<String, u64>) -> IndexMap<u64, String> {
    let mut counts: IndexMap<u64, (usize, String)> = IndexMap::new();
    for (id, &sig) in signatures {
        let entry = counts.entry(sig).or_insert((0, String::new()));
        entry.0 += 1;
        if entry.0 == 1 {
            entry.1 = id.clone();
        }
    }
    counts
        .into_iter()
        .filter(|(_, (count, _))| *count == 1)
        .map(|(sig, (_, id))| (sig, id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, op_type: &str, inputs: &[&str]) -> FxNode {
        FxNode {
            id: id.to_string(),
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            attrs: Default::default(),
            out_shape: Vec::new(),
            out_dtype: None,
            source_file: None,
            source_line: None,
        }
    }

    fn sigs(nodes: &[FxNode]) -> IndexMap<String, u64> {
        FxGraph::from_nodes(nodes).signatures(&CommutativitySet::standard())
    }

    /// THE key property: swapping the operands of a non-commutative op changes the swapped node's
    /// signature, so the diff can see it. (Operands distinguished by placeholder arg index.)
    #[test]
    fn test_noncommutative_sub_swap_differs() {
        let base = [
            node("a", PLACEHOLDER, &[]),
            node("b", PLACEHOLDER, &[]),
            node("s", "aten.sub.Tensor", &["a", "b"]),
        ];
        let head = [
            node("a", PLACEHOLDER, &[]),
            node("b", PLACEHOLDER, &[]),
            node("s", "aten.sub.Tensor", &["b", "a"]),
        ];
        assert_ne!(sigs(&base)["s"], sigs(&head)["s"]);
    }

    /// The counterpart: swapping the operands of a commutative op leaves the signature unchanged,
    /// so the diff stays silent.
    #[test]
    fn test_commutative_add_swap_same() {
        let base = [
            node("a", PLACEHOLDER, &[]),
            node("b", PLACEHOLDER, &[]),
            node("s", "aten.add.Tensor", &["a", "b"]),
        ];
        let head = [
            node("a", PLACEHOLDER, &[]),
            node("b", PLACEHOLDER, &[]),
            node("s", "aten.add.Tensor", &["b", "a"]),
        ];
        assert_eq!(sigs(&base)["s"], sigs(&head)["s"]);
    }

    /// Signatures use no node names, so renaming every node leaves them unchanged — a pure rename
    /// is matchable, not mistaken for a rewrite.
    #[test]
    fn test_signatures_are_rename_invariant() {
        let original = [
            node("a", PLACEHOLDER, &[]),
            node("r", "aten.relu.default", &["a"]),
        ];
        let renamed = [
            node("x", PLACEHOLDER, &[]),
            node("y", "aten.relu.default", &["x"]),
        ];
        let a = sigs(&original);
        let b = sigs(&renamed);
        assert_eq!(a["a"], b["x"]);
        assert_eq!(a["r"], b["y"]);
    }

    /// Same graph in, identical signatures out across repeated computation (determinism, D10).
    #[test]
    fn test_signatures_deterministic() {
        let nodes = [
            node("a", PLACEHOLDER, &[]),
            node("b", "aten.abs.default", &["a"]),
            node("c", "aten.sub.Tensor", &["a", "b"]),
        ];
        assert_eq!(sigs(&nodes), sigs(&nodes));
    }

    /// Anchors are the signatures unique in both graphs and equal; the unchanged inputs of an
    /// add-a-node edit anchor, while the node whose consumer set changed does not.
    #[test]
    fn test_extract_anchors_pairs_unique_matches() {
        let base = [
            node("a", PLACEHOLDER, &[]),
            node("b", "aten.abs.default", &["a"]),
        ];
        let head = [
            node("a2", PLACEHOLDER, &[]),
            node("b2", "aten.abs.default", &["a2"]),
            node("c2", "aten.relu.default", &["b2"]),
        ];
        let anchors = extract_anchors(
            &FxGraph::from_nodes(&base),
            &FxGraph::from_nodes(&head),
            &CommutativitySet::standard(),
        );
        // The placeholder is unchanged on both sides (same op, same consumer op_type) and anchors;
        // `b`'s consumer set changed (gained relu in head), so its signature shifts and it does not.
        assert!(anchors
            .iter()
            .any(|x| x.base_id == "a" && x.head_id == "a2"));
        assert!(anchors.iter().all(|x| x.base_id != "b"));
    }

    /// A signature shared by two nodes is ambiguous, so it is not used as an anchor.
    #[test]
    fn test_duplicate_signature_is_not_an_anchor() {
        // Two structurally identical relu-of-input chains: their nodes collide pairwise.
        let g = [
            node("p", PLACEHOLDER, &[]),
            node("r1", "aten.relu.default", &["p"]),
            node("r2", "aten.relu.default", &["p"]),
        ];
        let graph = FxGraph::from_nodes(&g);
        let anchors = extract_anchors(&graph, &graph, &CommutativitySet::standard());
        // r1 and r2 share a signature (same op, same input, same empty consumer set), so neither
        // anchors; only the unique placeholder does.
        assert!(anchors.iter().any(|x| x.base_id == "p"));
        assert!(anchors
            .iter()
            .all(|x| x.base_id != "r1" && x.base_id != "r2"));
    }
}
