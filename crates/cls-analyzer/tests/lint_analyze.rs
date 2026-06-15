//! Integration tests for the Tool 4 lint analyzer: join candidate findings with the correctness
//! database and apply the severity-escalation rules.

use cls_analyzer::lint::{analyze, render, Format};
use cls_correctness_db::PatternDb;
use cls_schema::ClsArtifact;

/// An artifact with one candidate lint finding, parameterized by the session's torch version and
/// the candidate's pattern. Built from JSON to avoid constructing a full `Session` by hand.
fn artifact(torch_version: &str, pattern: &str) -> ClsArtifact {
    let json = format!(
        r#"{{"schema_version":"0.5.0",
            "session":{{"id":"00000000-0000-4000-8000-000000000000",
                        "timestamp":"2026-06-12T00:00:00Z","torch_version":"{torch_version}",
                        "redaction_policy":"default-strict"}},
            "lint_findings":[{{"finding_id":"f1","pattern_category":"{pattern}","severity":"medium",
                               "source_location":{{"file":"model.py","line_start":47}}}}]}}"#
    );
    serde_json::from_str(&json).expect("artifact parses")
}

const DB_TOML: &str = r#"
[[patterns]]
name = "in_place_op_on_alias"
minimal_repro = "y = x.expand(2, 3); y[0] = 1"
pytorch_issue = "https://github.com/pytorch/pytorch/issues/114302"
workaround = "y = x.expand(2, 3).clone()"
li_et_al_section = "§3.2.3"
fixed_in_version = "2.5"
[patterns.fixtures]
positive = "p"
negative = "n"
"#;

fn db() -> PatternDb {
    PatternDb::from_toml(DB_TOML).expect("db loads")
}

#[test]
fn unfixed_for_user_torch_escalates_to_high() {
    // torch 2.4 < fixed_in 2.5: the bug still bites -> high, with the DB's evidence attached.
    let report = analyze(&artifact("2.4.0", "in_place_op_on_alias"), &db());
    assert_eq!(report.findings.len(), 1);
    let f = &report.findings[0];
    assert_eq!(f.severity, "high");
    assert_eq!(
        f.reference_issue.as_ref().and_then(|r| r.url.as_deref()),
        Some("https://github.com/pytorch/pytorch/issues/114302")
    );
    assert_eq!(f.workaround.as_deref(), Some("y = x.expand(2, 3).clone()"));
    assert_eq!(f.li_et_al_section.as_deref(), Some("§3.2.3"));
}

#[test]
fn fixed_for_user_torch_downgrades_to_info() {
    // torch 2.6 >= fixed_in 2.5: already fixed -> info.
    let report = analyze(&artifact("2.6.0", "in_place_op_on_alias"), &db());
    assert_eq!(report.findings[0].severity, "info");
}

#[test]
fn pattern_not_in_db_is_dropped() {
    // A candidate with no DB entry has no evidence (no issue link) -> not a finding.
    let report = analyze(&artifact("2.4.0", "unknown_pattern"), &db());
    assert!(report.findings.is_empty());
}

#[test]
fn markdown_lists_finding_with_severity_and_evidence() {
    let md = render(
        &analyze(&artifact("2.4.0", "in_place_op_on_alias"), &db()),
        Format::Markdown,
    );
    assert!(md.contains("Compile Lint"), "{md}");
    assert!(md.contains("high"), "{md}");
    assert!(md.contains("in_place_op_on_alias"), "{md}");
    assert!(md.contains("issues/114302"), "{md}");
    assert!(md.contains("model.py:47"), "{md}");
}

#[test]
fn markdown_empty_says_so() {
    let md = render(
        &analyze(&artifact("2.4.0", "unknown_pattern"), &db()),
        Format::Markdown,
    );
    assert!(md.contains("No lint findings"), "{md}");
}

#[test]
fn json_round_trips() {
    let report = analyze(&artifact("2.4.0", "in_place_op_on_alias"), &db());
    let json = render(&report, Format::Json);
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert_eq!(v["findings"][0]["severity"], "high");
    assert_eq!(v["findings"][0]["pattern_category"], "in_place_op_on_alias");
}
