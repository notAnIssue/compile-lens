//! Tool 4 lint analyzer — join candidate lint findings with the correctness database and escalate
//! severity (design notes).
//!
//! The Python scanner (Layers 1 + 2) writes candidate findings into the artifact's `lint_findings`;
//! this analyzer turns each into a real finding by looking its pattern up in the correctness
//! database (`cls-correctness-db`), attaching the four ADR-013 evidence items, and setting severity:
//!
//! - in the database, and **not** fixed for the session's torch version → `high`;
//! - in the database, and already fixed for the session's torch version → `info`;
//! - **not** in the database (no issue link / no evidence) → dropped — never a finding.
//!
//! That last rule is the lint-not-oracle discipline: a finding without a cited issue does not ship.

use cls_correctness_db::{Pattern, PatternDb};
use cls_schema::{ClsArtifact, LintFinding, ReferenceIssue};
use serde::Serialize;

/// The analyzed lint findings — each enriched with severity and the database's evidence.
#[derive(Debug, Clone, Serialize)]
pub struct LintReport {
    pub findings: Vec<LintFinding>,
}

/// Analyze the artifact's candidate lint findings against the correctness database.
pub fn analyze(artifact: &ClsArtifact, db: &PatternDb) -> LintReport {
    let user_version = parse_version(&artifact.session.torch_version);
    let findings = artifact
        .lint_findings
        .iter()
        .filter_map(|candidate| {
            // No database entry → no evidence → not a finding (lint-not-oracle).
            let pattern = db.lookup(&candidate.pattern_category)?;
            Some(enrich(candidate, pattern, user_version))
        })
        .collect();
    LintReport { findings }
}

/// Build the enriched finding: candidate's identity + location, the database's evidence, and the
/// escalated severity.
fn enrich(
    candidate: &LintFinding,
    pattern: &Pattern,
    user_version: Option<(u32, u32)>,
) -> LintFinding {
    let fixed = is_fixed(pattern, user_version);
    LintFinding {
        finding_id: candidate.finding_id.clone(),
        pattern_category: candidate.pattern_category.clone(),
        severity: if fixed { "info" } else { "high" }.to_string(),
        source_location: candidate.source_location.clone(),
        trigger_pattern: candidate.trigger_pattern.clone(),
        li_et_al_section: pattern.li_et_al_section.clone(),
        reference_issue: Some(ReferenceIssue {
            url: Some(pattern.pytorch_issue.clone()),
            fixed_in_version: pattern.fixed_in_version.clone(),
        }),
        workaround: Some(pattern.workaround.clone()),
        confidence: candidate.confidence.clone(),
        applies_to_user_torch_version: Some(!fixed),
    }
}

/// Whether the pattern's issue is fixed for the user's torch version. Unknown fix version, or an
/// unparseable version on either side, is treated as **not fixed** (the conservative, high side).
fn is_fixed(pattern: &Pattern, user_version: Option<(u32, u32)>) -> bool {
    match (
        pattern.fixed_in_version.as_deref().and_then(parse_version),
        user_version,
    ) {
        (Some(fixed_in), Some(user)) => user >= fixed_in,
        _ => false,
    }
}

/// Parse a `major.minor` prefix from a version string (`"2.6.0"` → `(2, 6)`); patch is ignored.
fn parse_version(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

/// Output format. `Text` is folded into `Markdown` by the CLI, so only two shapes exist here.
#[derive(Clone, Copy, Debug)]
pub enum Format {
    Markdown,
    Json,
}

/// Render the report.
pub fn render(report: &LintReport, format: Format) -> String {
    match format {
        Format::Json => serde_json::to_string_pretty(report).expect("LintReport serializes"),
        Format::Markdown => render_markdown(report),
    }
}

fn render_markdown(report: &LintReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "## Compile Lint");

    if report.findings.is_empty() {
        let _ = writeln!(out, "\nNo lint findings.");
        return out;
    }

    for f in &report.findings {
        let _ = writeln!(out);
        let loc = f
            .source_location
            .as_ref()
            .and_then(|s| {
                s.file.as_deref().map(|file| match s.line_start {
                    Some(line) => format!("{file}:{line}"),
                    None => file.to_string(),
                })
            })
            .unwrap_or_else(|| "<unknown>".to_string());
        let _ = writeln!(out, "### [{}] `{}` — {loc}", f.severity, f.pattern_category);
        if let Some(url) = f.reference_issue.as_ref().and_then(|r| r.url.as_deref()) {
            let _ = writeln!(out, "- Issue: {url}");
        }
        if let Some(workaround) = &f.workaround {
            let _ = writeln!(out, "- Workaround: {workaround}");
        }
        if let Some(section) = &f.li_et_al_section {
            let _ = writeln!(out, "- Li et al. {section}");
        }
    }
    out
}
