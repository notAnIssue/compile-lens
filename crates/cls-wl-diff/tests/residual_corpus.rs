//! Validates residual classification against the committed diff fixture corpus
//! (`tests/fixtures/diff/`). Each fixture runs the full pipeline (signature → anchors → expansion
//! → residual) and the classification is checked against what the fixture encodes: `add_node`
//! yields one added, `remove_node` one removed, `modify_attr` one modified.

use std::fs;
use std::path::{Path, PathBuf};

use cls_schema::ClsArtifact;
use cls_wl_diff::{
    classify_residual, expand_from_anchors, extract_anchors, CommutativitySet, FxGraph,
    ResidualClassification,
};

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/diff")
        .join(name)
}

/// Load `<fixture>/<side>.cls.json` and build its single compiled graph.
fn graph(fixture: &str, side: &str) -> FxGraph {
    let path = fixture_dir(fixture).join(side);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let artifact: ClsArtifact =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    FxGraph::from_nodes(&artifact.compiled_graphs[0].nodes)
}

/// Run the full diff pipeline over a fixture's base/head.
fn classify(fixture: &str) -> ResidualClassification {
    let before = graph(fixture, "base.cls.json");
    let after = graph(fixture, "head.cls.json");
    let comm = CommutativitySet::standard();
    let anchors = extract_anchors(&before, &after, &comm);
    let matching = expand_from_anchors(&before, &after, &anchors, &comm);
    classify_residual(&before, &after, &matching, &comm)
}

#[test]
fn test_add_node_fixture_classifies_one_added() {
    let result = classify("add_node");
    assert_eq!(result.added.len(), 1, "add_node: {:?}", result.added);
    assert!(result.removed.is_empty());
    assert!(result.modified.is_empty());
}

#[test]
fn test_remove_node_fixture_classifies_one_removed() {
    let result = classify("remove_node");
    assert_eq!(result.removed.len(), 1, "remove_node: {:?}", result.removed);
    assert!(result.added.is_empty());
    assert!(result.modified.is_empty());
}

#[test]
fn test_modify_attr_fixture_classifies_one_modified() {
    let result = classify("modify_attr");
    assert_eq!(
        result.modified.len(),
        1,
        "modify_attr: {:?}",
        result.modified
    );
    assert!(result.added.is_empty());
    assert!(result.removed.is_empty());
}
