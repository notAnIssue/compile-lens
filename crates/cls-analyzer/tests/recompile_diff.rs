//! Tests for session-to-session recompile diff (ADR-033, the regression mode).
//!
//! Builds two in-memory artifacts (baseline + head) from real Mode-A guard text and checks the
//! added / grown / removed classification + the regression-anchored suggestions.

use cls_analyzer::recompile::RecompileDiff;
use cls_analyzer::recompile_diff::diff_recompiles;
use cls_schema::{ClsArtifact, FailedGuard, Recompilation, RedactionPolicy, Session};

fn session() -> Session {
    Session {
        id: "t".into(),
        timestamp: "2026-01-01T00:00:00Z".into(),
        torch_version: "2.6.0".into(),
        redaction_policy: RedactionPolicy::DefaultStrict,
        triton_version: None,
        cuda_version: None,
        python_version: None,
        gpu_name: None,
        host: None,
        command: None,
        duration_ms: None,
        iteration_count: None,
        git_sha: None,
        workspace_fingerprint: None,
        rank: None,
        world_size: None,
        collector_versions: Default::default(),
        compile_config: None,
        env_snapshot: None,
        extra: Default::default(),
    }
}

fn rec(id: &str, expr: &str, prev: &str, new: &str) -> Recompilation {
    Recompilation {
        recompilation_id: id.into(),
        compiled_function_id: Some("f".into()),
        trigger_reason: Some("guard_failure".into()),
        failed_guard: Some(FailedGuard {
            guard_id: Some("0/0".into()),
            expression: Some(expr.into()),
            previous_value: Some(prev.into()),
            new_value: Some(new.into()),
        }),
        occurred_at_step: Some(1),
        wall_clock_ms: None,
        source_location: None,
    }
}

fn artifact(recs: Vec<Recompilation>) -> ClsArtifact {
    let mut a = ClsArtifact::new(session());
    a.recompilations = recs;
    a
}

fn diff(base: Vec<Recompilation>, head: Vec<Recompilation>) -> RecompileDiff {
    diff_recompiles(&artifact(base), &artifact(head))
}

const SIZE0: &str = "tensor 'x' size mismatch at index 0";
const DTYPE: &str = "tensor 'x' dtype mismatch";

#[test]
fn added_cluster_is_a_new_axis_in_head() {
    // baseline has only a size recompile; head also recompiles on dtype -> dtype is ADDED.
    let d = diff(
        vec![rec("0/1", SIZE0, "4", "8")],
        vec![
            rec("0/1", SIZE0, "4", "8"),
            rec("0/2", DTYPE, "Float", "Double"),
        ],
    );
    assert_eq!(d.added.len(), 1);
    assert_eq!(d.added[0].category, "dtype");
    assert!(d.grown.is_empty());
    assert!(d.removed.is_empty());
}

#[test]
fn grown_cluster_recompiles_more_in_head() {
    // same axis, but head recompiles it 3× vs baseline 1× -> GROWN (1 -> 3).
    let d = diff(
        vec![rec("0/1", SIZE0, "4", "8")],
        vec![
            rec("0/1", SIZE0, "4", "8"),
            rec("0/2", SIZE0, "8", "16"),
            rec("0/3", SIZE0, "16", "32"),
        ],
    );
    assert!(d.added.is_empty());
    assert_eq!(d.grown.len(), 1);
    assert_eq!(d.grown[0].base_count, 1);
    assert_eq!(d.grown[0].cluster.count, 3);
    assert_eq!(d.grown[0].cluster.axis.as_deref(), Some("x[0]"));
}

#[test]
fn removed_cluster_is_fixed_in_head() {
    // baseline recompiled on dtype; head no longer does -> REMOVED (fixed).
    let d = diff(
        vec![
            rec("0/1", SIZE0, "4", "8"),
            rec("0/2", DTYPE, "Float", "Double"),
        ],
        vec![rec("0/1", SIZE0, "4", "8")],
    );
    assert_eq!(d.removed.len(), 1);
    assert_eq!(d.removed[0].category, "dtype");
    assert!(d.added.is_empty() && d.grown.is_empty());
}

#[test]
fn identical_sessions_have_empty_diff() {
    let same = || vec![rec("0/1", SIZE0, "4", "8")];
    let d = diff(same(), same());
    assert_eq!(d, RecompileDiff::default());
}

#[test]
fn added_cluster_yields_regression_anchored_suggestion() {
    // The headline demo: a PR that introduces a recompile axis -> a regression-framed fix.
    let d = diff(
        vec![],
        vec![rec("0/1", SIZE0, "4", "8"), rec("0/2", SIZE0, "8", "16")],
    );
    assert_eq!(d.added.len(), 1);
    assert_eq!(d.suggestions.len(), 1);
    let s = &d.suggestions[0];
    assert!(s.text.contains("introduced"), "got: {}", s.text);
    assert!(s.text.contains("mark_dynamic(x, 0)")); // the fix is still axis-precise
    assert!(s.evidence.contains("ADDED vs baseline"));
}

#[test]
fn grown_cluster_suggestion_shows_the_delta() {
    let d = diff(
        vec![rec("0/1", SIZE0, "4", "8")],
        vec![rec("0/1", SIZE0, "4", "8"), rec("0/2", SIZE0, "8", "16")],
    );
    let s = &d.suggestions[0];
    assert!(s.text.contains("worsened") && s.text.contains("1 → 2"));
    assert!(s.text.contains("mark_dynamic(x, 0)"));
}

#[test]
fn other_category_regression_yields_no_suggestion() {
    // An unrecognized guard regresses, but `other` never produces a suggestion (N2) — it still
    // shows up as added, just without advice.
    let d = diff(vec![], vec![rec("0/1", "n == 1", "1", "2")]);
    assert_eq!(d.added.len(), 1);
    assert!(d.suggestions.is_empty());
}
