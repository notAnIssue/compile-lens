//! `diff_graphs` end-to-end against the committed synthetic fixture corpus (`tests/fixtures/diff/`).
//! Each fixture's base/head run through the full pipeline; the structured diff is checked against
//! the fixture's `expected.diff.json` oracle (added / removed / modified by node id).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use cls_schema::ClsArtifact;
use cls_wl_diff::{diff_graphs, FxGraph, IrGraphDiff};
use serde_json::Value;

/// The nine synthetic fixtures (the two real captured pairs are deferred until the collector can
/// emit node-level graphs; see the fixture corpus README).
const SYNTHETIC: &[&str] = &[
    "add_node",
    "remove_node",
    "modify_attr",
    "operand_swap_sub",
    "operand_swap_div",
    "operand_swap_matmul",
    "commutative_add",
    "variable_rename",
    "fusion_lost",
];

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/diff")
        .join(name)
}

fn graph(fixture: &str, side: &str) -> FxGraph {
    let path = fixture_dir(fixture).join(side);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let artifact: ClsArtifact =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    FxGraph::from_nodes(&artifact.compiled_graphs[0].nodes)
}

fn diff(fixture: &str) -> IrGraphDiff {
    diff_graphs(
        &graph(fixture, "base.cls.json"),
        &graph(fixture, "head.cls.json"),
    )
}

fn oracle_ids(fixture: &str, field: &str) -> BTreeSet<String> {
    let path = fixture_dir(fixture).join("expected.diff.json");
    let value: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    value[field]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Every synthetic fixture's diff matches its oracle's added / removed / modified sets.
#[test]
fn test_all_synthetic_fixtures_match_their_oracle() {
    for &fixture in SYNTHETIC {
        let d = diff(fixture);
        let added: BTreeSet<String> = d.added.iter().cloned().collect();
        let removed: BTreeSet<String> = d.removed.iter().cloned().collect();
        // The oracle records modified node ids; our pairs share the id on both sides for these
        // fixtures (rename aside, which has no modifications), so compare on the base id.
        let modified: BTreeSet<String> = d.modified.iter().map(|(b, _)| b.clone()).collect();

        assert_eq!(added, oracle_ids(fixture, "added"), "{fixture}: added");
        assert_eq!(
            removed,
            oracle_ids(fixture, "removed"),
            "{fixture}: removed"
        );
        assert_eq!(
            modified,
            oracle_ids(fixture, "modified"),
            "{fixture}: modified"
        );
    }
}

/// The two operand-order release-blocker fixtures, called out explicitly: the non-commutative
/// swaps are reported as modified, the commutative swap is silent.
#[test]
fn test_operand_order_blocker_fixtures() {
    for fixture in [
        "operand_swap_sub",
        "operand_swap_div",
        "operand_swap_matmul",
    ] {
        let d = diff(fixture);
        assert_eq!(
            d.modified.len(),
            1,
            "{fixture}: must report one modified node"
        );
        assert!(
            d.added.is_empty() && d.removed.is_empty(),
            "{fixture}: must not add/remove"
        );
    }
    let add = diff("commutative_add");
    assert!(
        add.added.is_empty() && add.removed.is_empty() && add.modified.is_empty(),
        "commutative_add: must be silent"
    );
}

/// `diff_graphs` is a pure function (D4): the same inputs give a byte-identical diff every time.
#[test]
fn test_diff_graphs_is_deterministic() {
    for &fixture in SYNTHETIC {
        assert_eq!(
            diff(fixture),
            diff(fixture),
            "{fixture}: diff must be deterministic"
        );
    }
}

/// Coverage and the ambiguity score stay within their defined ranges across the corpus.
#[test]
fn test_quality_numbers_in_range() {
    for &fixture in SYNTHETIC {
        let d = diff(fixture);
        assert!(
            (0.0..=1.0).contains(&d.match_coverage),
            "{fixture}: coverage out of range"
        );
        assert!(
            (0.0..=1.0).contains(&d.anchor_uniqueness_ratio),
            "{fixture}: uniqueness out of range"
        );
        for (base, _, confidence) in &d.matched {
            assert!(
                (0.0..=1.0).contains(confidence),
                "{fixture}/{base}: confidence out of range"
            );
        }
    }
}
