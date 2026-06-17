//! Tests for the hero HTML report renderer.

use cls_schema::ClsArtifact;

fn from_json(json: &str) -> ClsArtifact {
    serde_json::from_str(json).expect("artifact parses")
}

/// Render with no extra inputs — the common case (no diff, no DB-escalated lint, no roofline).
fn plain(artifact: &ClsArtifact) -> String {
    cls_report::render(artifact, &cls_report::ReportInputs::default())
}

/// A committed example artifact, addressed relative to this crate's manifest dir.
fn example(name: &str) -> ClsArtifact {
    let path = format!(
        "{}/../../schema/examples/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    from_json(&std::fs::read_to_string(path).expect("read example"))
}

const MINIMAL_SESSION: &str = r#"{"schema_version":"0.5.0","session":{"id":"s1",
"timestamp":"2026-01-01T00:00:00Z","torch_version":"2.6.0","redaction_policy":"default-strict",
"gpu_name":"A100-SXM-80GB"}}"#;

#[test]
fn renders_a_self_contained_offline_document() {
    let html = plain(&from_json(MINIMAL_SESSION));
    assert!(html.starts_with("<!DOCTYPE html>"), "missing doctype");
    assert!(html.contains("<style>"), "CSS must be inlined");
    // Offline: no external/CDN references anywhere in the document.
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "no external URLs allowed"
    );
    assert!(html.contains("Session metadata"));
    assert!(html.contains("2.6.0"), "torch version shown");
    assert!(html.contains("A100-SXM-80GB"), "gpu shown");
}

#[test]
fn a_cache_stable_session_says_so() {
    // minimal.cls.json has no recompilations.
    let html = plain(&example("minimal.cls.json"));
    assert!(html.contains("Recompile summary"));
    assert!(html.contains("cache-stable"));
}

#[test]
fn renders_the_recompilation_from_the_full_example() {
    // full.cls.json carries one recompilation.
    let html = plain(&example("full.cls.json"));
    assert!(html.contains("Recompile summary"));
    assert!(html.contains("recompilation(s)"));
    assert!(html.contains("<table>"));
    assert!(html.contains("Raw artifacts"));
}

#[test]
fn user_controlled_strings_are_escaped() {
    // A torch_version carrying an injection must render as inert text, never live markup.
    let html = plain(&from_json(
        r#"{"schema_version":"0.5.0","session":{"id":"s1","timestamp":"t",
        "torch_version":"<script>alert(1)</script>","redaction_policy":"default-strict"}}"#,
    ));
    assert!(
        !html.contains("<script>alert(1)</script>"),
        "injection must not survive verbatim"
    );
    assert!(html.contains("&lt;script&gt;"), "must be escaped");
}

#[test]
fn divergence_section_surfaces_a_nan_as_the_headline_signal() {
    let html = plain(&from_json(
        r#"{"schema_version":"0.5.0","session":{"id":"s1","timestamp":"t",
        "torch_version":"2.6.0","redaction_policy":"default-strict"},
        "divergences":[{"divergence_id":"d1","first_divergent_layer":"layers.3.mlp",
        "max_abs_diff":"NaN","num_layers_compared":12,"rtol":1e-3,"atol":1e-5,
        "suggested_cause":"inductor fusion of the residual add"}]}"#,
    ));
    assert!(html.contains("Divergence (eager vs compiled)"));
    assert!(html.contains("layers.3.mlp"));
    // The NaN sentinel surfaces as the headline "compiled output is NaN", not a dropped value.
    assert!(html.contains("compiled output is NaN"), "{html}");
}

#[test]
fn fusion_section_renders_the_crown_jewel() {
    let html = plain(&from_json(
        r#"{"schema_version":"0.5.0","session":{"id":"s1","timestamp":"t",
        "torch_version":"2.6.0","redaction_policy":"default-strict"},
        "fusion_opportunities":[{"pattern_id":"A","shape":{"m":4096,"n":4096,"k0":4096,"k1":4096,
        "dtype":"bfloat16"},"baseline_hbm_bytes":1.0e8,"fused_hbm_bytes":5.0e7,
        "estimated_speedup":2.04,"suggested_kernel":"epilogue_kit.ops.gemm_rmsnorm",
        "confidence":"high"}]}"#,
    ));
    assert!(html.contains("Fusion opportunities (CODA)"));
    assert!(html.contains("2.04×"), "speedup shown: {html}");
    assert!(html.contains("epilogue_kit.ops.gemm_rmsnorm"));
    assert!(html.contains("4096×4096×4096×4096"));
}

#[test]
fn empty_sections_say_so_rather_than_placeholder() {
    let html = plain(&from_json(MINIMAL_SESSION));
    assert!(html.contains("No divergence findings"));
    assert!(html.contains("No algebraic fusion opportunities"));
}

/// A committed fixture under the repo's top-level `tests/fixtures/`.
fn repo_fixture(rel: &str) -> ClsArtifact {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    from_json(&std::fs::read_to_string(path).expect("read fixture"))
}

#[test]
fn cache_stability_section_renders_a_silently_stale_finding() {
    // listing2 = Li et al. Listing 2 — a silently-stale cache (state drifts, graph reused, output
    // frozen).
    let html = plain(&repo_fixture(
        "tests/fixtures/cache_stability/listing2.cls.json",
    ));
    assert!(html.contains("Cache stability"));
    assert!(
        html.contains("drifted attrs"),
        "findings table should render: {html}"
    );
}

#[test]
fn cache_stability_section_reports_stable_when_no_iterations() {
    let html = plain(&from_json(MINIMAL_SESSION));
    assert!(html.contains("Cache stability"));
    assert!(html.contains("No silently-stale-cache"));
}

#[test]
fn ir_diff_section_is_absent_without_a_baseline() {
    let html = plain(&from_json(MINIMAL_SESSION));
    assert!(!html.contains("IR Diff"), "no baseline -> no diff section");
}

#[test]
fn ir_diff_section_leads_with_the_structural_change() {
    let diff = cls_report::IrGraphDiff {
        added: vec!["mul_2".into()],
        removed: vec![],
        modified: vec![("to_1".into(), "to_1b".into())],
        matched: vec![("to_1".into(), "to_1b".into(), 0.91)],
        match_coverage: 0.9,
        anchor_uniqueness_ratio: 0.8,
    };
    let html = cls_report::render(
        &from_json(MINIMAL_SESSION),
        &cls_report::ReportInputs {
            diff: Some(&diff),
            ..Default::default()
        },
    );
    assert!(html.contains("IR Diff (base → head)"));
    assert!(
        html.contains("<strong>1</strong> added"),
        "change counts: {html}"
    );
    assert!(html.contains("to_1b"), "modified head id shown");
    assert!(html.contains("0.91"), "match confidence shown");
    assert!(
        html.contains("90%") && html.contains("80%"),
        "quality gauges shown"
    );
}

#[test]
fn ir_diff_section_says_clean_when_graphs_are_identical() {
    let diff = cls_report::IrGraphDiff {
        added: vec![],
        removed: vec![],
        modified: vec![],
        matched: vec![],
        match_coverage: 1.0,
        anchor_uniqueness_ratio: 1.0,
    };
    let html = cls_report::render(
        &from_json(MINIMAL_SESSION),
        &cls_report::ReportInputs {
            diff: Some(&diff),
            ..Default::default()
        },
    );
    assert!(html.contains("No structural change"));
}

#[test]
fn lint_section_degrades_to_candidates_without_a_db() {
    // No --db -> the renderer shows the artifact's raw candidates and how to escalate.
    let html = plain(&from_json(
        r#"{"schema_version":"0.5.0","session":{"id":"s1","timestamp":"t",
        "torch_version":"2.6.0","redaction_policy":"default-strict"},
        "lint_findings":[{"finding_id":"l1","pattern_category":"inplace_on_alias","severity":"",
        "source_location":{"file":"model.py","line_start":42},"trigger_pattern":"x.add_(y)",
        "confidence":"medium"}]}"#,
    ));
    assert!(html.contains("Lint (Tool 4)"));
    assert!(html.contains("candidate pattern(s)"));
    assert!(html.contains("--db"), "should say how to escalate");
    assert!(html.contains("inplace_on_alias"));
    assert!(html.contains("model.py:42"));
}

#[test]
fn lint_section_shows_escalated_findings_with_a_db() {
    let report = cls_analyzer::lint::LintReport {
        findings: vec![cls_schema::LintFinding {
            finding_id: "l1".into(),
            pattern_category: "inplace_on_alias".into(),
            severity: "high".into(),
            source_location: Some(cls_schema::SourceRange {
                file: Some("model.py".into()),
                line_start: Some(42),
                line_end: None,
                code_excerpt: None,
            }),
            trigger_pattern: None,
            li_et_al_section: None,
            reference_issue: Some(cls_schema::ReferenceIssue {
                url: Some("https://github.com/pytorch/pytorch/issues/12345".into()),
                fixed_in_version: None,
            }),
            workaround: Some("clone before the in-place op".into()),
            confidence: Some("high".into()),
            applies_to_user_torch_version: Some(true),
        }],
    };
    let html = cls_report::render(
        &from_json(MINIMAL_SESSION),
        &cls_report::ReportInputs {
            lint: Some(&report),
            ..Default::default()
        },
    );
    assert!(html.contains("Lint (Tool 4)"));
    assert!(html.contains("inplace_on_alias"));
    assert!(html.contains(">high<"), "severity badge: {html}");
    assert!(html.contains("model.py:42"));
    assert!(html.contains("issues/12345"), "issue link shown");
}

#[test]
fn roofline_section_shows_per_kernel_predictions() {
    // Run the real analyzer over a fixture that carries kernel features, then render its report.
    let artifact = repo_fixture("tests/fixtures/roofline/sample.cls.json");
    let report = cls_analyzer::roofline::analyze(&artifact, "A100-SXM-80GB", 8).expect("known gpu");
    let html = cls_report::render(
        &artifact,
        &cls_report::ReportInputs {
            roofline: Some(&report),
            ..Default::default()
        },
    );
    assert!(html.contains("Roofline (Tool 5)"));
    assert!(html.contains("A100-SXM-80GB"), "gpu shown");
    assert!(
        html.contains("memory") || html.contains("compute"),
        "a bound type should render: {html}"
    );
}

#[test]
fn roofline_section_absent_says_not_computed() {
    let html = plain(&from_json(MINIMAL_SESSION));
    assert!(html.contains("Roofline (Tool 5)"));
    assert!(html.contains("Roofline not computed"));
}

/// An escalated lint report carrying a single finding whose issue URL is `url`.
fn lint_report_with_issue_url(url: &str) -> cls_analyzer::lint::LintReport {
    cls_analyzer::lint::LintReport {
        findings: vec![cls_schema::LintFinding {
            finding_id: "l1".into(),
            pattern_category: "inplace_on_alias".into(),
            severity: "high".into(),
            source_location: None,
            trigger_pattern: None,
            li_et_al_section: None,
            reference_issue: Some(cls_schema::ReferenceIssue {
                url: Some(url.into()),
                fixed_in_version: None,
            }),
            workaround: None,
            confidence: None,
            applies_to_user_torch_version: Some(true),
        }],
    }
}

fn render_with_lint(report: &cls_analyzer::lint::LintReport) -> String {
    cls_report::render(
        &from_json(MINIMAL_SESSION),
        &cls_report::ReportInputs {
            lint: Some(report),
            ..Default::default()
        },
    )
}

#[test]
fn issue_url_on_the_allowlist_becomes_a_link() {
    let report = lint_report_with_issue_url("https://github.com/pytorch/pytorch/issues/12345");
    let html = render_with_lint(&report);
    assert!(
        html.contains("<a href=\"https://github.com/pytorch/pytorch/issues/12345\">"),
        "allowlisted issue should link: {html}"
    );
}

#[test]
fn issue_url_off_the_allowlist_stays_inert() {
    // An arbitrary host, a non-https scheme, and a javascript: URI must never become anchors.
    for url in [
        "javascript:alert(1)",
        "https://evil.example/pwn",
        "http://github.com/pytorch/x",
    ] {
        let html = render_with_lint(&lint_report_with_issue_url(url));
        assert!(!html.contains("<a href"), "must not link {url}: {html}");
        assert!(
            html.contains(&cls_report::esc(url)),
            "url still shown inert: {url}"
        );
    }
}
