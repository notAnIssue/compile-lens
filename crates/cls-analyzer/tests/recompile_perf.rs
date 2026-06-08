//! Hard performance gate for Tool 1's analyzer (phase-level AC: ≥1000 recompiles < 1s).
//!
//! This is the deterministic CI gate — a fast, single-shot timing assertion that fails loudly
//! on a perf regression without running the full statistical bench (`benches/recompile_summary.rs`,
//! which is the measurement / scaling instrument run via `cargo bench`). The 1s bound is the
//! phase target; it holds with a large margin even in an unoptimized test build, so the
//! assertion is robust against CI-runner noise rather than flaky.

use std::time::Instant;

use cls_analyzer::recompile::analyze;
use cls_schema::ClsArtifact;
use serde_json::json;

/// `n` size-mismatch recompiles across 4 dynamic axes — same shape the bench uses.
fn synthetic_artifact(n: usize) -> ClsArtifact {
    let recompiles: Vec<_> = (0..n)
        .map(|i| {
            let (dim, prev, new) = (i % 4, 8 + i, 9 + i);
            json!({
                "recompilation_id": format!("rec_{i}"),
                "trigger_reason": "guard_failure",
                "failed_guard": {
                    "guard_id": format!("g_{i}"),
                    "expression": format!(
                        "tensor 'x' size mismatch at index {dim}. expected {prev}, actual {new}"
                    ),
                    "previous_value": prev.to_string(),
                    "new_value": new.to_string(),
                }
            })
        })
        .collect();
    let artifact = json!({
        "schema_version": "0.5.0",
        "session": {
            "id": "00000000-0000-4000-8000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "torch_version": "2.6.0",
            "redaction_policy": "default-strict"
        },
        "recompilations": recompiles,
    });
    serde_json::from_value(artifact).expect("valid synthetic artifact")
}

#[test]
fn analyze_1000_events_under_one_second() {
    let artifact = synthetic_artifact(1_000);
    let start = Instant::now();
    let findings = analyze(&artifact).expect("analyze succeeds");
    let elapsed = start.elapsed();

    assert_eq!(findings.total_recompilations, 1_000);
    assert!(
        elapsed.as_secs_f64() < 1.0,
        "1000-event analyze took {elapsed:?}, exceeds the 1s phase target"
    );
}

#[test]
fn analyze_scales_to_ten_thousand_events() {
    // Scaling sanity: 10k events is one O(n) pass; if it blew past 1s the cost is super-linear.
    let artifact = synthetic_artifact(10_000);
    let start = Instant::now();
    let findings = analyze(&artifact).expect("analyze succeeds");
    let elapsed = start.elapsed();

    assert_eq!(findings.total_recompilations, 10_000);
    // 4 axes -> 4 clusters regardless of event count (groups by (category, axis)).
    assert_eq!(findings.guard_categories.len(), 4);
    assert!(
        elapsed.as_secs_f64() < 1.0,
        "10000-event analyze took {elapsed:?} — clustering may have gone super-linear"
    );
}
