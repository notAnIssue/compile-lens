//! Validates the WL-signature against the committed diff fixture corpus (`tests/fixtures/diff/`).
//!
//! The corpus encodes the operand-order spec as data; this checks the signature actually delivers
//! it on those real fixtures: a non-commutative operand swap must change the swapped node's
//! signature, and a commutative one must not. The swapped node in each operand-order fixture is
//! `n2`.

use std::fs;
use std::path::{Path, PathBuf};

use cls_schema::ClsArtifact;
use cls_wl_diff::{CommutativitySet, FxGraph};

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/diff")
        .join(name)
}

/// Signature of node `id` in the single compiled graph of `<fixture>/<side>.cls.json`.
fn node_signature(fixture: &str, side: &str, id: &str) -> u64 {
    let path = fixture_dir(fixture).join(side);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let artifact: ClsArtifact =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let nodes = &artifact.compiled_graphs[0].nodes;
    let signatures = FxGraph::from_nodes(nodes).signatures(&CommutativitySet::standard());
    signatures[id]
}

/// Every non-commutative operand-swap fixture: the swapped node's signature must differ
/// base-vs-head, so the diff sees the change.
#[test]
fn test_noncommutative_swaps_change_signature() {
    for fixture in [
        "operand_swap_sub",
        "operand_swap_div",
        "operand_swap_matmul",
    ] {
        assert_ne!(
            node_signature(fixture, "base.cls.json", "n2"),
            node_signature(fixture, "head.cls.json", "n2"),
            "{fixture}: swapped node n2 must have a different signature in head"
        );
    }
}

/// The commutative-add swap fixture: the swapped node's signature must be identical base-vs-head,
/// so the diff stays silent.
#[test]
fn test_commutative_swap_preserves_signature() {
    assert_eq!(
        node_signature("commutative_add", "base.cls.json", "n2"),
        node_signature("commutative_add", "head.cls.json", "n2"),
        "commutative_add: swapped node n2 must keep its signature (silent)"
    );
}
