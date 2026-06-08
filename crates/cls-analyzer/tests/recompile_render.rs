//! Renderer tests for Tool 1 (`recompile_render`).
//!
//! Built from hand-constructed findings (not from a `.cls.json`) so the test pins the
//! *rendering*, decoupled from clustering/suggestion logic upstream. The Markdown shapes are
//! locked against an inline expected literal — the literal *is* the snapshot, reviewable in
//! the test and drift-detecting (any wording/layout change fails), without pulling a snapshot
//! crate's transitive dependencies into a deliberately small project (ADR-015). JSON is
//! asserted valid + round-tripping the structured fields; `text` is asserted to carry no
//! Markdown furniture.

use cls_analyzer::recompile::{
    GrownCluster, GuardCategory, RecompileDiff, RecompileFindings, Suggestion, ValueTransition,
};
use cls_analyzer::recompile_render::{render_diff, render_findings, Format};

/// A representative multi-cluster summary: a size cluster (with an axis + value
/// transitions + a suggestion) and a dtype cluster.
fn sample_findings() -> RecompileFindings {
    RecompileFindings {
        total_recompilations: 47,
        guard_categories: vec![
            GuardCategory {
                category: "size".into(),
                count: 32,
                canonical_guard: "tensor '<t>' size mismatch at index <n>".into(),
                axis: Some("x[0]".into()),
                observed_values: vec![
                    ValueTransition {
                        previous: Some("8".into()),
                        new: Some("16".into()),
                    },
                    ValueTransition {
                        previous: Some("16".into()),
                        new: Some("24".into()),
                    },
                ],
            },
            GuardCategory {
                category: "dtype".into(),
                count: 12,
                canonical_guard: "tensor '<t>' dtype mismatch".into(),
                axis: Some("w".into()),
                observed_values: vec![ValueTransition {
                    previous: Some("torch.float16".into()),
                    new: Some("torch.bfloat16".into()),
                }],
            },
        ],
        top_suggestions: vec![Suggestion {
            text: "`x` dim 0 keeps changing size — mark it dynamic: `mark_dynamic(x, 0)`.".into(),
            evidence: "size cluster on x[0] — 32 recompile(s) (values 8→16, 16→24)".into(),
        }],
    }
}

fn sample_diff() -> RecompileDiff {
    let added = vec![GuardCategory {
        category: "size".into(),
        count: 8,
        canonical_guard: "tensor '<t>' size mismatch at index <n>".into(),
        axis: Some("x[0]".into()),
        observed_values: vec![ValueTransition {
            previous: Some("4".into()),
            new: Some("8".into()),
        }],
    }];
    let grown = vec![GrownCluster {
        cluster: GuardCategory {
            category: "dtype".into(),
            count: 12,
            canonical_guard: "tensor '<t>' dtype mismatch".into(),
            axis: Some("w".into()),
            observed_values: vec![],
        },
        base_count: 4,
    }];
    RecompileDiff {
        added,
        grown,
        removed: vec![],
        suggestions: vec![Suggestion {
            text: "This change introduced 8 recompile(s) on a new axis — mark `x` dim 0 dynamic."
                .into(),
            evidence: "ADDED vs baseline · size cluster on x[0] — 8 recompile(s)".into(),
        }],
    }
}

// The two `*_matches_snapshot` literals below are written flush-left on purpose: they must
// match the rendered bytes exactly (leading tree-indent included), so no source indentation
// can be added inside the literal.

#[test]
fn render_markdown_matches_snapshot() {
    let expected = "## Recompile Summary

Total recompilations: 47
Triggered by: 2 distinct guard categories
  ├ size · `x[0]` (32 events — values 8→16, 16→24)
  └ dtype · `w` (12 events — values torch.float16→torch.bfloat16)

### Top suggestions

1. `x` dim 0 keeps changing size — mark it dynamic: `mark_dynamic(x, 0)`.
   _evidence: size cluster on x[0] — 32 recompile(s) (values 8→16, 16→24)_
";
    assert_eq!(
        render_findings(&sample_findings(), Format::Markdown),
        expected
    );
}

#[test]
fn render_diff_markdown_matches_snapshot() {
    let expected = "## Recompile Diff (vs baseline)

Added: 1 · Grown: 1 · Removed: 0

### Added — new recompiles this change introduced
  └ size · `x[0]` (8 events — values 4→8)

### Grown — recompiled more than baseline
  └ dtype · `w` (12 events) — was 4 in baseline

### Removed — no longer recompiling
  (none)

### Suggestions

1. This change introduced 8 recompile(s) on a new axis — mark `x` dim 0 dynamic.
   _evidence: ADDED vs baseline · size cluster on x[0] — 8 recompile(s)_
";
    assert_eq!(render_diff(&sample_diff(), Format::Markdown), expected);
}

#[test]
fn render_json_validates_and_roundtrips() {
    let json = render_findings(&sample_findings(), Format::Json);
    let value: serde_json::Value = serde_json::from_str(&json).expect("output must be valid JSON");
    // The structured fields survive serialization (machine-readable contract).
    assert_eq!(value["total_recompilations"], 47);
    assert_eq!(value["guard_categories"][0]["category"], "size");
    assert_eq!(value["guard_categories"][0]["axis"], "x[0]");
    assert_eq!(
        value["guard_categories"][0]["observed_values"][0]["new"],
        "16"
    );
    assert!(value["top_suggestions"][0]["text"].is_string());
}

#[test]
fn render_diff_json_validates() {
    let json = render_diff(&sample_diff(), Format::Json);
    let value: serde_json::Value =
        serde_json::from_str(&json).expect("diff output must be valid JSON");
    assert_eq!(value["added"][0]["category"], "size");
    assert_eq!(value["grown"][0]["base_count"], 4);
    assert!(value["removed"]
        .as_array()
        .expect("removed is array")
        .is_empty());
}

#[test]
fn render_text_is_minimal_no_markdown_furniture() {
    let text = render_findings(&sample_findings(), Format::Text);
    assert!(text.contains("Recompile Summary"));
    assert!(text.contains("Total recompilations: 47"));
    assert!(text.contains("size"));
    // The renderer's own structural furniture is plain ASCII: no Markdown headers, and the
    // cluster lines it builds use ASCII `->` arrows (Unicode in *suggestion content* is the
    // analyzer's wording, which `text` passes through verbatim rather than mangling).
    assert!(
        !text.contains("## "),
        "text must not contain Markdown headers: {text}"
    );
    assert!(
        !text.contains("### "),
        "text must not contain Markdown headers: {text}"
    );
    assert!(
        text.contains("8->16"),
        "renderer-built value arrows should be ASCII: {text}"
    );
}

#[test]
fn empty_findings_render_cleanly_in_all_formats() {
    let empty = RecompileFindings::default();
    assert!(render_findings(&empty, Format::Markdown).contains("No recompilations recorded"));
    assert!(render_findings(&empty, Format::Text).contains("No recompilations recorded"));
    // Empty findings still produce valid JSON (an object with a zero count).
    let json = render_findings(&empty, Format::Json);
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(value["total_recompilations"], 0);
}

#[test]
fn empty_diff_reports_no_difference() {
    let empty = RecompileDiff::default();
    assert!(render_diff(&empty, Format::Markdown).contains("No recompile differences"));
    assert!(render_diff(&empty, Format::Text).contains("No recompile differences"));
}
