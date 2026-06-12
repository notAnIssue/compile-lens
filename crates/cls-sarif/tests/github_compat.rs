//! GitHub Code Scanning compatibility for the SARIF 2.1.0 output.
//!
//! Code Scanning ingests SARIF 2.1.0; these assert the structure it requires — `version`,
//! `runs[].tool.driver.name`, and `results[]` each carrying `ruleId`, `level`, `message.text`, and
//! a physical location — and the severity → level mapping.

use cls_sarif::to_sarif;
use cls_schema::{LintFinding, ReferenceIssue, SourceRange};

fn finding(severity: &str) -> LintFinding {
    LintFinding {
        finding_id: "f1".into(),
        pattern_category: "in_place_op_on_alias".into(),
        severity: severity.into(),
        source_location: Some(SourceRange {
            file: Some("model.py".into()),
            line_start: Some(47),
            line_end: None,
            code_excerpt: None,
        }),
        trigger_pattern: None,
        li_et_al_section: Some("§3.2.3".into()),
        reference_issue: Some(ReferenceIssue {
            url: Some("https://github.com/pytorch/pytorch/issues/114302".into()),
            fixed_in_version: Some("2.5".into()),
        }),
        workaround: Some("y = x.expand(2, 3).clone()".into()),
        confidence: None,
        applies_to_user_torch_version: Some(true),
    }
}

fn json_of(findings: &[LintFinding]) -> serde_json::Value {
    let log = to_sarif(findings, "compile-lens", "0.5.0");
    serde_json::from_str(&serde_json::to_string(&log).expect("serializes")).expect("valid json")
}

#[test]
fn has_the_structure_code_scanning_requires() {
    let v = json_of(&[finding("high")]);
    assert_eq!(v["version"], "2.1.0");
    assert!(v["$schema"].as_str().unwrap().contains("sarif-2.1.0"));
    assert_eq!(v["runs"][0]["tool"]["driver"]["name"], "compile-lens");
    assert_eq!(v["runs"][0]["tool"]["driver"]["version"], "0.5.0");

    let r = &v["runs"][0]["results"][0];
    assert_eq!(r["ruleId"], "in_place_op_on_alias");
    assert_eq!(r["level"], "error"); // high -> error
    assert!(r["message"]["text"].is_string());
    let phys = &r["locations"][0]["physicalLocation"];
    assert_eq!(phys["artifactLocation"]["uri"], "model.py");
    assert_eq!(phys["region"]["startLine"], 47);
}

#[test]
fn high_is_error_info_is_note() {
    assert_eq!(
        json_of(&[finding("high")])["runs"][0]["results"][0]["level"],
        "error"
    );
    assert_eq!(
        json_of(&[finding("info")])["runs"][0]["results"][0]["level"],
        "note"
    );
}

#[test]
fn the_rule_is_declared_in_the_driver_with_its_issue_link() {
    let v = json_of(&[finding("high")]);
    let rule = &v["runs"][0]["tool"]["driver"]["rules"][0];
    assert_eq!(rule["id"], "in_place_op_on_alias");
    assert!(
        rule["helpUri"].as_str().unwrap().contains("issues/114302"),
        "rule should carry the issue as helpUri"
    );
}

#[test]
fn repeated_pattern_is_declared_once_but_reported_each_time() {
    let v = json_of(&[finding("high"), finding("high")]);
    assert_eq!(
        v["runs"][0]["tool"]["driver"]["rules"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(v["runs"][0]["results"].as_array().unwrap().len(), 2);
}

#[test]
fn empty_findings_is_a_valid_empty_run() {
    let v = json_of(&[]);
    assert_eq!(v["version"], "2.1.0");
    assert_eq!(v["runs"][0]["results"].as_array().unwrap().len(), 0);
}
