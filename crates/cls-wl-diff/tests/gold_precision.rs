//! Gold-corpus precision benchmark (design discipline D7 invariant).
//!
//! A corpus of synthetic graph edits whose correct node mapping is known by construction (the
//! "gold" mapping). For each case we run `diff_graphs` and check how many of the matches it
//! reports are correct. **Precision** is correct matches over total reported matches, aggregated
//! across the corpus — it measures how often the matcher is right when it claims two nodes
//! correspond. The corpus has 20+ entries and the asserted floor is 0.90.
//!
//! Inline (no fixture files) so the gold mappings live next to the graphs they describe and the
//! corpus is self-contained. Runs in CI via `cargo test --workspace`.

use cls_schema::FxNode;
use cls_wl_diff::{diff_graphs, FxGraph};

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

/// A placeholder followed by a chain of single-input ops. `<prefix>_p` is the input, `<prefix>_n0`,
/// `<prefix>_n1`, … the chain. Distinct ids per prefix let a rename be expressed as a gold mapping.
fn chain(prefix: &str, ops: &[&str]) -> Vec<FxNode> {
    let mut nodes = vec![node(&format!("{prefix}_p"), "placeholder", &[])];
    let mut prev = format!("{prefix}_p");
    for (i, op) in ops.iter().enumerate() {
        let id = format!("{prefix}_n{i}");
        nodes.push(node(&id, op, &[&prev]));
        prev = id;
    }
    nodes
}

/// One gold case: two graphs and the node pairs that *correctly* correspond between them.
struct Gold {
    name: String,
    base: Vec<FxNode>,
    head: Vec<FxNode>,
    correct: Vec<(String, String)>,
}

fn ids(nodes: &[FxNode]) -> Vec<String> {
    nodes.iter().map(|n| n.id.clone()).collect()
}

/// A pure rename: head is the same chain under a different prefix; every node correctly maps to
/// its positional counterpart.
fn rename_case(ops: &[&str]) -> Gold {
    let base = chain("a", ops);
    let head = chain("b", ops);
    let correct = ids(&base).into_iter().zip(ids(&head)).collect();
    Gold {
        name: format!("rename_{}", ops.join("-")),
        base,
        head,
        correct,
    }
}

/// Append one op to the chain (same prefix, so the original nodes keep their ids): the original
/// nodes all correctly map to themselves; the appended node is added (not in the gold mapping).
fn add_case(ops: &[&str], extra: &str) -> Gold {
    let base = chain("c", ops);
    let mut head_ops = ops.to_vec();
    head_ops.push(extra);
    let head = chain("c", &head_ops);
    let correct = ids(&base).into_iter().map(|id| (id.clone(), id)).collect();
    Gold {
        name: format!("add_{extra}_to_{}", ops.join("-")),
        base,
        head,
        correct,
    }
}

/// Drop the last op (mirror of add): the surviving nodes map to themselves.
fn remove_case(ops: &[&str]) -> Gold {
    let base = chain("c", ops);
    let head = chain("c", &ops[..ops.len() - 1]);
    let correct = ids(&head).into_iter().map(|id| (id.clone(), id)).collect();
    Gold {
        name: format!("remove_last_of_{}", ops.join("-")),
        base,
        head,
        correct,
    }
}

/// A non-commutative operand swap with distinguishable operands; the swapped node still correctly
/// corresponds to itself (it is modified, but matched).
fn swap_case(op: &str) -> Gold {
    let base = vec![
        node("a", "placeholder", &[]),
        node("b", "aten.abs.default", &["a"]),
        node("n", op, &["a", "b"]),
    ];
    let head = vec![
        node("a", "placeholder", &[]),
        node("b", "aten.abs.default", &["a"]),
        node("n", op, &["b", "a"]),
    ];
    let correct = vec![
        ("a".into(), "a".into()),
        ("b".into(), "b".into()),
        ("n".into(), "n".into()),
    ];
    Gold {
        name: format!("swap_{op}"),
        base,
        head,
        correct,
    }
}

/// An attribute-only change; the node still correctly corresponds to itself (modified, matched).
fn modify_case(name: &str, op: &str) -> Gold {
    let base = vec![
        node("p", "placeholder", &[]),
        node_attr("m", op, &["p"], "k", 1),
    ];
    let head = vec![
        node("p", "placeholder", &[]),
        node_attr("m", op, &["p"], "k", 2),
    ];
    let correct = vec![("p".into(), "p".into()), ("m".into(), "m".into())];
    Gold {
        name: name.to_string(),
        base,
        head,
        correct,
    }
}

fn corpus() -> Vec<Gold> {
    vec![
        // 6 renames of varying length/op
        rename_case(&["aten.relu.default"]),
        rename_case(&["aten.relu.default", "aten.abs.default"]),
        rename_case(&["aten.relu.default", "aten.abs.default", "aten.neg.default"]),
        rename_case(&["aten.abs.default", "aten.neg.default", "aten.relu.default"]),
        rename_case(&[
            "aten.relu.default",
            "aten.abs.default",
            "aten.neg.default",
            "aten.relu.default",
        ]),
        rename_case(&[
            "aten.neg.default",
            "aten.relu.default",
            "aten.abs.default",
            "aten.neg.default",
            "aten.relu.default",
        ]),
        // 4 add-a-node
        add_case(&["aten.relu.default"], "aten.abs.default"),
        add_case(
            &["aten.relu.default", "aten.abs.default"],
            "aten.neg.default",
        ),
        add_case(
            &["aten.relu.default", "aten.abs.default", "aten.neg.default"],
            "aten.relu.default",
        ),
        add_case(
            &["aten.abs.default", "aten.neg.default"],
            "aten.relu.default",
        ),
        // 4 remove-a-node
        remove_case(&["aten.relu.default", "aten.abs.default"]),
        remove_case(&["aten.relu.default", "aten.abs.default", "aten.neg.default"]),
        remove_case(&["aten.abs.default", "aten.neg.default", "aten.relu.default"]),
        remove_case(&[
            "aten.relu.default",
            "aten.abs.default",
            "aten.neg.default",
            "aten.relu.default",
        ]),
        // 4 operand swaps (non-commutative)
        swap_case("aten.sub.Tensor"),
        swap_case("aten.div.Tensor"),
        swap_case("aten.matmul.default"),
        swap_case("aten.pow.Tensor_Tensor"),
        // 4 attribute modifications
        modify_case("modify_relu", "aten.relu.default"),
        modify_case("modify_abs", "aten.abs.default"),
        modify_case("modify_neg", "aten.neg.default"),
        modify_case("modify_gelu", "aten.gelu.default"),
    ]
}

#[test]
fn test_gold_corpus_precision_clears_threshold() {
    let cases = corpus();
    assert!(
        cases.len() >= 20,
        "gold corpus needs 20+ entries, has {}",
        cases.len()
    );

    let mut correct_total = 0usize;
    let mut reported_total = 0usize;
    for case in &cases {
        let d = diff_graphs(
            &FxGraph::from_nodes(&case.base),
            &FxGraph::from_nodes(&case.head),
        );
        let gold: std::collections::BTreeSet<(String, String)> =
            case.correct.iter().cloned().collect();
        let correct = d
            .matched
            .iter()
            .filter(|(b, h, _)| gold.contains(&(b.clone(), h.clone())))
            .count();
        if correct != d.matched.len() {
            println!(
                "{}: {correct}/{} matches correct",
                case.name,
                d.matched.len()
            );
        }
        correct_total += correct;
        reported_total += d.matched.len();
    }

    let precision = correct_total as f64 / reported_total as f64;
    println!("gold precision = {precision:.3} ({correct_total}/{reported_total} matches correct over {} cases)", cases.len());
    assert!(
        precision >= 0.90,
        "gold-corpus precision {precision:.3} is below the 0.90 release-blocker floor; \
         the matcher is producing wrong matches — investigate, do not lower the floor"
    );
}
