//! Tests for Tool 1 guard clustering — the describe → attribute behaviour (ADR-032).
//!
//! Inputs are built in-memory from the real Mode-A guard text shapes
//! (`tensor 'x' size mismatch at index 0`, …) so the clustering is exercised against the same
//! strings the collector produces, without coupling a Rust test to a committed `.cls.json`.

use cls_analyzer::recompile::{self, GuardCategory};
use cls_schema::{ClsArtifact, FailedGuard, Recompilation, RedactionPolicy, Session};

fn session() -> Session {
    Session {
        id: "test".into(),
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

/// A recompile carrying a guard with an `expr` and `previous -> new` values.
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

fn artifact(recompilations: Vec<Recompilation>) -> ClsArtifact {
    let mut a = ClsArtifact::new(session());
    a.recompilations = recompilations;
    a
}

fn analyze(recompilations: Vec<Recompilation>) -> Vec<GuardCategory> {
    recompile::analyze(&artifact(recompilations))
        .expect("analyze never errors")
        .guard_categories
}

#[test]
fn cluster_single_category() {
    // Storm shape: many recompiles, same axis -> one cluster.
    let clusters = analyze(vec![
        rec("0/1", "tensor 'x' size mismatch at index 0", "1", "2"),
        rec("0/2", "tensor 'x' size mismatch at index 0", "2", "4"),
        rec("0/3", "tensor 'x' size mismatch at index 0", "4", "8"),
    ]);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].category, "size");
    assert_eq!(clusters[0].count, 3);
    assert_eq!(clusters[0].axis.as_deref(), Some("x[0]"));
}

#[test]
fn cluster_multi_category() {
    // mixed_guards shape: size / dtype / stride -> three distinct clusters.
    let clusters = analyze(vec![
        rec("0/1", "tensor 'x' size mismatch at index 0", "8", "16"),
        rec("0/2", "tensor 'x' dtype mismatch", "Float", "Double"),
        rec("0/3", "tensor 'x' stride mismatch at index 0", "4", "1"),
    ]);
    let cats: Vec<&str> = clusters.iter().map(|c| c.category.as_str()).collect();
    assert_eq!(clusters.len(), 3);
    assert!(cats.contains(&"size") && cats.contains(&"dtype") && cats.contains(&"stride"));
}

#[test]
fn cluster_canonical_form_extraction() {
    let clusters = analyze(vec![rec(
        "0/1",
        "tensor 'x' size mismatch at index 0",
        "4",
        "8",
    )]);
    let c = &clusters[0];
    assert_eq!(c.canonical_guard, "tensor '<t>' size mismatch at index <n>");
    assert_eq!(c.axis.as_deref(), Some("x[0]"));
    // dtype has no index -> axis is the whole tensor, distinct canonical template
    let dtype = analyze(vec![rec(
        "0/1",
        "tensor 'w' dtype mismatch",
        "Float",
        "Double",
    )]);
    assert_eq!(dtype[0].canonical_guard, "tensor '<t>' dtype mismatch");
    assert_eq!(dtype[0].axis.as_deref(), Some("w"));
}

#[test]
fn cluster_axis_level_separation() {
    // Two dims of the same tensor going dynamic are two actionable axes -> two clusters.
    let clusters = analyze(vec![
        rec("0/1", "tensor 'x' size mismatch at index 0", "8", "16"),
        rec("0/2", "tensor 'x' size mismatch at index 1", "4", "8"),
    ]);
    assert_eq!(clusters.len(), 2);
    let axes: Vec<Option<&str>> = clusters.iter().map(|c| c.axis.as_deref()).collect();
    assert!(axes.contains(&Some("x[0]")) && axes.contains(&Some("x[1]")));
}

#[test]
fn cluster_observed_value_transitions_deduped() {
    let clusters = analyze(vec![
        rec("0/1", "tensor 'x' size mismatch at index 0", "4", "8"),
        rec("0/2", "tensor 'x' size mismatch at index 0", "4", "8"), // duplicate transition
        rec("0/3", "tensor 'x' size mismatch at index 0", "8", "16"),
    ]);
    assert_eq!(clusters[0].count, 3);
    // distinct transitions only: (4->8) and (8->16)
    let transitions: Vec<(Option<&str>, Option<&str>)> = clusters[0]
        .observed_values
        .iter()
        .map(|v| (v.previous.as_deref(), v.new.as_deref()))
        .collect();
    assert_eq!(
        transitions,
        vec![(Some("4"), Some("8")), (Some("8"), Some("16"))]
    );
}

#[test]
fn cluster_other_category_literal_canonicalization() {
    // An unrecognized guard groups by its literal-canonicalized form.
    let clusters = analyze(vec![
        rec("0/1", "self.batch_size == 32", "32", "24"),
        rec("0/2", "self.batch_size == 16", "16", "8"),
    ]);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].category, "other");
    assert_eq!(clusters[0].canonical_guard, "self.batch_size == <int>");
    assert!(clusters[0].axis.is_none());
}

#[test]
fn cluster_malformed_guard_skipped() {
    // A recompile with no guard / no expression is counted in the total but contributes no
    // cluster (must not crash or invent a bogus cluster).
    let mut no_guard = rec("0/1", "ignored", "a", "b");
    no_guard.failed_guard = None;
    let mut empty_expr = rec("0/2", "", "a", "b");
    if let Some(g) = empty_expr.failed_guard.as_mut() {
        g.expression = Some(String::new());
    }
    let good = rec("0/3", "tensor 'x' size mismatch at index 0", "4", "8");

    let findings = recompile::analyze(&artifact(vec![no_guard, empty_expr, good]))
        .expect("analyze never errors");
    assert_eq!(findings.total_recompilations, 3); // all counted
    assert_eq!(findings.guard_categories.len(), 1); // only the parseable one clustered
}

#[test]
fn cluster_is_deterministic() {
    let build = || {
        vec![
            rec("0/1", "tensor 'x' size mismatch at index 0", "8", "16"),
            rec("0/2", "tensor 'y' dtype mismatch", "Float", "Double"),
            rec("0/3", "tensor 'x' size mismatch at index 0", "16", "32"),
        ]
    };
    assert_eq!(analyze(build()), analyze(build()));
    // most-frequent cluster first (size@x[0], count 2) before the dtype singleton
    let clusters = analyze(build());
    assert_eq!(clusters[0].count, 2);
    assert_eq!(clusters[0].axis.as_deref(), Some("x[0]"));
}
