//! Structural integrity of the production correctness database (`data/correctness_db.toml`).
//!
//! The catalogue gate: load the real database and check every entry is well-formed — the four
//! ADR-013 items present *and* non-empty, a real PyTorch issue URL, unique names, and (for
//! operator-family patterns) a usable detector. That the four items are *present* is already
//! enforced by `from_toml` (required fields); this adds the content checks a type can't express.
//! Behavioral fixture validation — positive fixture detected, negative clean — runs on the Python
//! scanner side, where detection lives (ADR-036).

use std::collections::HashSet;

use cls_correctness_db::PatternDb;

fn production_db() -> PatternDb {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/correctness_db.toml"
    );
    let toml = std::fs::read_to_string(path).expect("read data/correctness_db.toml");
    PatternDb::from_toml(&toml).expect("production database loads (all ADR-013 items present)")
}

#[test]
fn database_is_non_empty() {
    assert!(!production_db().is_empty());
}

#[test]
fn names_are_unique() {
    // lookup() returns the first match by name, so duplicates would silently shadow.
    let db = production_db();
    let mut seen = HashSet::new();
    for p in db.patterns() {
        assert!(
            seen.insert(p.name.as_str()),
            "duplicate pattern name: {}",
            p.name
        );
    }
}

#[test]
fn every_pattern_has_non_empty_evidence() {
    for p in production_db().patterns() {
        assert!(
            !p.minimal_repro.trim().is_empty(),
            "{}: empty minimal_repro",
            p.name
        );
        assert!(
            !p.workaround.trim().is_empty(),
            "{}: empty workaround",
            p.name
        );
        assert!(
            !p.fixtures.positive.trim().is_empty(),
            "{}: empty positive fixture",
            p.name
        );
        assert!(
            !p.fixtures.negative.trim().is_empty(),
            "{}: empty negative fixture",
            p.name
        );
    }
}

#[test]
fn every_issue_is_a_real_pytorch_issue_url() {
    // ADR-013 item 2 is evidence the bug is reported, not a guess — so it must be a pytorch issue.
    for p in production_db().patterns() {
        assert!(
            p.pytorch_issue
                .starts_with("https://github.com/pytorch/pytorch/issues/"),
            "{}: pytorch_issue is not a PyTorch issue URL: {}",
            p.name,
            p.pytorch_issue
        );
    }
}

#[test]
fn operator_detectors_are_well_formed() {
    for p in production_db().patterns() {
        let Some(det) = &p.detector else { continue };
        assert!(
            !det.operator.trim().is_empty(),
            "{}: detector has empty operator",
            p.name
        );
        assert!(!det.params.is_empty(), "{}: detector has no params", p.name);
        assert!(
            det.params.iter().all(|param| !param.trim().is_empty()),
            "{}: detector has an empty param name",
            p.name
        );
    }
}
