//! Guards the compile-diff fixture corpus (`tests/fixtures/diff/`).
//!
//! The diff algorithm does not exist yet; these tests assert only that the corpus is
//! well-formed and that the spec it encodes is internally consistent, so a malformed
//! fixture fails here rather than silently misleading a later algorithm test. They check
//! three things: every `base`/`head` artifact deserializes through the real `cls-schema`
//! bindings and actually carries node-level graph structure; every `expected.diff.json`
//! has the oracle's required shape; and the two operand-order key cases say what they must.

use std::fs;
use std::path::{Path, PathBuf};

use cls_schema::ClsArtifact;
use serde_json::Value;

/// `tests/fixtures/diff`, resolved from this crate's manifest (workspace root is two up).
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/diff")
}

/// Every scenario subdirectory (skips the README and any stray files).
fn scenarios() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(corpus_dir())
        .expect("corpus dir should exist")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

fn parse_artifact(path: &Path) -> ClsArtifact {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{} should deserialize as a .cls.json: {e}", path.display()))
}

fn read_json(path: &Path) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{} should be valid JSON: {e}", path.display()))
}

/// The corpus ships at least the nine synthetic scenarios; a smaller count means a fixture
/// was dropped (the real captured pairs are a separate, later addition).
#[test]
fn test_corpus_has_synthetic_scenarios() {
    assert!(
        scenarios().len() >= 9,
        "expected >= 9 synthetic scenarios, found {}",
        scenarios().len()
    );
}

/// Every base/head artifact must parse through the real bindings and carry a non-empty
/// node-level graph — the whole point of the corpus is to feed node-level diffing.
#[test]
fn test_every_base_and_head_carries_a_node_graph() {
    for dir in scenarios() {
        for side in ["base.cls.json", "head.cls.json"] {
            let path = dir.join(side);
            let artifact = parse_artifact(&path);
            let graph = artifact
                .compiled_graphs
                .first()
                .unwrap_or_else(|| panic!("{} should have a compiled graph", path.display()));
            assert!(
                !graph.nodes.is_empty(),
                "{} should carry node-level structure (nodes[])",
                path.display()
            );
        }
    }
}

/// Every oracle has the qualitative shape downstream tests rely on: the three
/// classification buckets, each an array.
#[test]
fn test_every_oracle_has_the_required_shape() {
    for dir in scenarios() {
        let path = dir.join("expected.diff.json");
        let oracle = read_json(&path);
        for field in ["added", "removed", "modified"] {
            assert!(
                oracle.get(field).and_then(Value::as_array).is_some(),
                "{} should have an array field `{field}`",
                path.display()
            );
        }
    }
}

/// The two operand-order cases are the corpus's reason to exist, so pin them explicitly:
/// a non-commutative swap must be reported, a commutative swap must stay silent.
#[test]
fn test_operand_order_key_cases_say_what_they_must() {
    let modified = |name: &str| -> Vec<String> {
        let oracle = read_json(&corpus_dir().join(name).join("expected.diff.json"));
        oracle["modified"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    };

    // Non-commutative sub: the swapped node must be flagged modified, not silently matched.
    assert_eq!(
        modified("operand_swap_sub"),
        vec!["n2"],
        "operand_swap_sub must report the swapped node as modified"
    );

    // Commutative add: reordering operands is semantically identical, so the diff is empty.
    let add = read_json(
        &corpus_dir()
            .join("commutative_add")
            .join("expected.diff.json"),
    );
    for field in ["added", "removed", "modified"] {
        assert!(
            add[field].as_array().unwrap().is_empty(),
            "commutative_add must stay silent, but `{field}` is non-empty"
        );
    }
}
