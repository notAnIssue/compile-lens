//! Tests for Tool 1 suggestion generation — the axis-precise "prescribe" step.
//!
//! Drives `analyze` end-to-end (recompiles -> clusters -> suggestions) on in-memory artifacts
//! built from the real Mode-A guard text, so the suggestions are exercised against the same
//! shapes the collector produces.

use cls_analyzer::recompile::{self, Suggestion};
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
    }
}

fn suggest(recompilations: Vec<Recompilation>) -> Vec<Suggestion> {
    let mut a = ClsArtifact::new(session());
    a.recompilations = recompilations;
    recompile::analyze(&a)
        .expect("analyze never errors")
        .top_suggestions
}

#[test]
fn suggest_mark_dynamic_is_axis_precise() {
    let s = suggest(vec![rec(
        "0/1",
        "tensor 'x' size mismatch at index 0",
        "4",
        "8",
    )]);
    assert_eq!(s.len(), 1);
    // the prescribe step names the exact axis, not a vague "consider mark_dynamic"
    assert!(
        s[0].text.contains("mark_dynamic(x, 0)"),
        "got: {}",
        s[0].text
    );
    assert!(!s[0].evidence.is_empty());
}

#[test]
fn suggest_dtype_fix() {
    let s = suggest(vec![rec(
        "0/1",
        "tensor 'x' dtype mismatch",
        "Float",
        "Double",
    )]);
    assert_eq!(s.len(), 1);
    assert!(s[0].text.contains("dtype") && s[0].text.contains("`x`"));
}

#[test]
fn suggest_contiguous_for_stride() {
    let s = suggest(vec![rec(
        "0/1",
        "tensor 'x' stride mismatch at index 0",
        "4",
        "1",
    )]);
    assert_eq!(s.len(), 1);
    assert!(s[0].text.contains("contiguous()"));
}

#[test]
fn no_suggestion_for_other_category() {
    // An unrecognized guard clusters into `other` (no axis) -> no confident suggestion (N2).
    let s = suggest(vec![rec("0/1", "n == 1", "1", "2")]);
    assert!(s.is_empty());
}

#[test]
fn suggestions_ranked_by_recompile_count() {
    // size@x[0] happens twice, dtype@x once -> the size suggestion ranks first.
    let s = suggest(vec![
        rec("0/1", "tensor 'x' size mismatch at index 0", "4", "8"),
        rec("0/2", "tensor 'x' dtype mismatch", "Float", "Double"),
        rec("0/3", "tensor 'x' size mismatch at index 0", "8", "16"),
    ]);
    assert_eq!(s.len(), 2);
    assert!(s[0].text.contains("mark_dynamic(x, 0)")); // most-frequent cluster first
    assert!(s[1].text.contains("dtype"));
}

#[test]
fn evidence_traces_to_the_cluster() {
    // N2: the evidence string carries the cluster's count + value transitions.
    let s = suggest(vec![
        rec("0/1", "tensor 'x' size mismatch at index 0", "4", "8"),
        rec("0/2", "tensor 'x' size mismatch at index 0", "8", "16"),
    ]);
    let ev = &s[0].evidence;
    assert!(ev.contains("size") && ev.contains("x[0]") && ev.contains("2 recompile"));
    assert!(ev.contains("4→8") && ev.contains("8→16"));
}

#[test]
fn empty_artifact_yields_no_suggestions() {
    assert!(suggest(vec![]).is_empty());
}
