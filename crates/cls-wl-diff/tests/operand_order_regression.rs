//! Operand-order regression — the v0.5.0 release blocker (design discipline D7).
//!
//! This is the single most important correctness guarantee of the compile-diff: swapping the
//! operands of a **non-commutative** op (`sub`, `div`, `matmul`, `pow`) is a real change and MUST
//! be reported as `modified`; swapping the operands of a **commutative** op (`add`, `mul`) is
//! semantically identical and MUST stay silent. Misjudging this is a silent diff bug a user could
//! essentially never catch — the numerics change while the tool reports "no difference".
//!
//! Kept in its own file, self-contained (inline graphs, no fixture coupling), so the guarantee is
//! maximally visible. **If any test here fails, the release is blocked.** It runs in CI through
//! `cargo test --workspace`.

use cls_schema::FxNode;
use cls_wl_diff::{diff_graphs, FxGraph};

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

/// Build a two-input op and its operand-swapped counterpart, then diff them. `a` and `b` are
/// distinct graph inputs (distinguished by argument position), so the swap is observable.
fn diff_operand_swap(op_type: &str) -> cls_wl_diff::IrGraphDiff {
    let base = [
        node("a", "placeholder", &[]),
        node("b", "placeholder", &[]),
        node("n", op_type, &["a", "b"]),
    ];
    let head = [
        node("a", "placeholder", &[]),
        node("b", "placeholder", &[]),
        node("n", op_type, &["b", "a"]), // operands swapped
    ];
    diff_graphs(&FxGraph::from_nodes(&base), &FxGraph::from_nodes(&head))
}

/// Non-commutative ops: an operand swap MUST be reported as exactly one modified node, and must
/// not be hidden behind an add/remove or, worse, dropped silently.
#[test]
fn test_noncommutative_operand_swap_is_modified() {
    for op in [
        "aten.sub.Tensor",
        "aten.div.Tensor",
        "aten.matmul.default",
        "aten.pow.Tensor_Tensor",
    ] {
        let d = diff_operand_swap(op);
        assert_eq!(
            d.modified.len(),
            1,
            "{op}: operand swap MUST be one modified node, got modified={:?} added={:?} removed={:?}",
            d.modified,
            d.added,
            d.removed
        );
        assert!(
            d.added.is_empty() && d.removed.is_empty(),
            "{op}: operand swap must be modified, not add/remove (added={:?} removed={:?})",
            d.added,
            d.removed
        );
    }
}

/// Commutative ops: an operand swap is semantically identical and MUST produce an empty diff.
#[test]
fn test_commutative_operand_swap_is_silent() {
    for op in ["aten.add.Tensor", "aten.mul.Tensor"] {
        let d = diff_operand_swap(op);
        assert!(
            d.added.is_empty() && d.removed.is_empty() && d.modified.is_empty(),
            "{op}: commutative operand swap MUST be silent, got added={:?} removed={:?} modified={:?}",
            d.added,
            d.removed,
            d.modified
        );
    }
}

/// Build a 3-operand non-commutative op and a counterpart with the FIRST operand stationary and the
/// other two swapped, then diff them. The stationary operand is the key: it lets neighborhood
/// expansion *match* the op (unlike a full 2-operand swap, where neither operand keeps its position
/// so the pair is only recovered later). This exercises the distinct path of a *matched* pair whose
/// input ORDER changed — the partial-reorder silent miss.
fn diff_partial_reorder(op_type: &str) -> cls_wl_diff::IrGraphDiff {
    let base = [
        node("a", "placeholder", &[]),
        node("b", "placeholder", &[]),
        node("c", "placeholder", &[]),
        node("n", op_type, &["a", "b", "c"]),
    ];
    let head = [
        node("a", "placeholder", &[]),
        node("b", "placeholder", &[]),
        node("c", "placeholder", &[]),
        node("n", op_type, &["a", "c", "b"]), // a stays at position 0; b and c swap
    ];
    diff_graphs(&FxGraph::from_nodes(&base), &FxGraph::from_nodes(&head))
}

/// Partial operand reorder on a non-commutative op MUST be reported as exactly one modified node.
/// This is the case the full-swap test cannot reach: `aten.addmm(bias, mat1, mat2)` — what every
/// biased `nn.Linear` lowers to — becoming `aten.addmm(bias, mat2, mat1)` keeps `bias` at position
/// 0, so expansion matches the op and the order change has to be caught at classification, not
/// silently dropped.
#[test]
fn test_noncommutative_partial_operand_reorder_is_modified() {
    for op in ["aten.addmm.default", "aten.baddbmm.default"] {
        let d = diff_partial_reorder(op);
        assert_eq!(
            d.modified.len(),
            1,
            "{op}: partial operand reorder MUST be one modified node, got modified={:?} added={:?} removed={:?}",
            d.modified,
            d.added,
            d.removed
        );
        assert!(
            d.added.is_empty() && d.removed.is_empty(),
            "{op}: partial reorder must be modified, not add/remove (added={:?} removed={:?})",
            d.added,
            d.removed
        );
    }
}

/// Guard the reorder check against over-firing: a 3-operand op with identical inputs in the SAME
/// order is a clean match and MUST stay silent (empty diff).
#[test]
fn test_identical_multi_operand_op_is_not_modified() {
    let nodes = [
        node("a", "placeholder", &[]),
        node("b", "placeholder", &[]),
        node("c", "placeholder", &[]),
        node("n", "aten.addmm.default", &["a", "b", "c"]),
    ];
    let d = diff_graphs(&FxGraph::from_nodes(&nodes), &FxGraph::from_nodes(&nodes));
    assert!(
        d.modified.is_empty() && d.added.is_empty() && d.removed.is_empty(),
        "identical multi-operand op must be an empty diff, got modified={:?} added={:?} removed={:?}",
        d.modified,
        d.added,
        d.removed
    );
}
