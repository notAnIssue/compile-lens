//! `cls-sarif` — SARIF 2.1.0 output for Tool 4 (`compile-lint`).
//!
//! [SARIF](https://sarifweb.azurewebsites.net/) is the JSON interchange format GitHub Code Scanning
//! ingests. This crate converts the lint findings (`cls_schema::LintFinding`) into the minimal
//! SARIF 2.1.0 subset Code Scanning requires: a `run` whose `tool.driver` declares each pattern as a
//! rule (with its PyTorch issue as the `helpUri`), and one `result` per finding carrying the rule
//! id, a `level`, a message, and a physical source location.
//!
//! Severity maps to the SARIF level Code Scanning understands: `high`/`critical` → `error`,
//! `medium`/`low` → `warning`, everything else (e.g. `info`) → `note`.

use std::collections::HashSet;

use cls_schema::LintFinding;
use serde::Serialize;

/// A SARIF 2.1.0 log (the top-level document).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifLog {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<Run>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Run {
    pub tool: Tool,
    pub results: Vec<SarifResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub driver: Driver,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Driver {
    pub name: String,
    pub version: String,
    pub rules: Vec<ReportingDescriptor>,
}

/// One rule in the driver — a lint pattern.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportingDescriptor {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_description: Option<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub help_uri: Option<String>,
}

/// One result — a finding at a location.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifResult {
    pub rule_id: String,
    pub level: String,
    pub message: Message,
    pub locations: Vec<Location>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Location {
    pub physical_location: PhysicalLocation,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalLocation {
    pub artifact_location: ArtifactLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<Region>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactLocation {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Region {
    pub start_line: u64,
}

const SCHEMA_URI: &str = "https://json.schemastore.org/sarif-2.1.0.json";

/// Convert lint findings into a SARIF 2.1.0 log for the named tool.
pub fn to_sarif(findings: &[LintFinding], tool_name: &str, tool_version: &str) -> SarifLog {
    let rules = collect_rules(findings);
    let results = findings.iter().map(result_for).collect();
    SarifLog {
        schema: SCHEMA_URI.to_string(),
        version: "2.1.0".to_string(),
        runs: vec![Run {
            tool: Tool {
                driver: Driver {
                    name: tool_name.to_string(),
                    version: tool_version.to_string(),
                    rules,
                },
            },
            results,
        }],
    }
}

/// Serialize findings straight to a pretty SARIF JSON string.
pub fn to_sarif_json(findings: &[LintFinding], tool_name: &str, tool_version: &str) -> String {
    serde_json::to_string_pretty(&to_sarif(findings, tool_name, tool_version))
        .expect("SarifLog serializes")
}

/// One rule per distinct pattern, in first-seen order, carrying its issue link as `helpUri`.
fn collect_rules(findings: &[LintFinding]) -> Vec<ReportingDescriptor> {
    let mut seen = HashSet::new();
    let mut rules = Vec::new();
    for f in findings {
        if seen.insert(f.pattern_category.as_str()) {
            rules.push(ReportingDescriptor {
                id: f.pattern_category.clone(),
                short_description: f.li_et_al_section.as_ref().map(|s| Message {
                    text: format!("Li et al. 2026 {s}"),
                }),
                help_uri: f.reference_issue.as_ref().and_then(|r| r.url.clone()),
            });
        }
    }
    rules
}

fn result_for(f: &LintFinding) -> SarifResult {
    SarifResult {
        rule_id: f.pattern_category.clone(),
        level: level_for(&f.severity),
        message: Message {
            text: message_for(f),
        },
        locations: locations_for(f),
    }
}

fn level_for(severity: &str) -> String {
    match severity {
        "high" | "critical" => "error",
        "medium" | "low" => "warning",
        _ => "note",
    }
    .to_string()
}

fn message_for(f: &LintFinding) -> String {
    let mut text = f.pattern_category.clone();
    if let Some(workaround) = &f.workaround {
        text.push_str(" — workaround: ");
        text.push_str(workaround);
    }
    text
}

fn locations_for(f: &LintFinding) -> Vec<Location> {
    let Some(source) = &f.source_location else {
        return Vec::new();
    };
    let Some(file) = &source.file else {
        return Vec::new();
    };
    vec![Location {
        physical_location: PhysicalLocation {
            artifact_location: ArtifactLocation { uri: file.clone() },
            region: source.line_start.map(|line| Region { start_line: line }),
        },
    }]
}
