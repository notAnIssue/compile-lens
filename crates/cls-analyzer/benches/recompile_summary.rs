//! Performance benchmark for Tool 1's analyzer (`recompile::analyze`).
//!
//! Phase-level target: a 1000-recompile session analyzes in well under a second, and the
//! cost scales ~linearly (clustering is one O(n) pass that parses each guard's text, groups
//! by `(category, axis)`, then sorts the clusters). We measure 100 / 1_000 / 10_000 events
//! and report throughput so a regression in the per-event cost shows up directly.
//!
//! The hard `< 1s` gate is a fast deterministic test (`tests/recompile_perf.rs`) so CI fails
//! on a perf regression without running the full statistical bench; this bench is the
//! measurement / scaling instrument.
//!
//! Run: `cargo bench --bench recompile_summary`.

use cls_analyzer::recompile::analyze;
use cls_schema::ClsArtifact;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::json;

/// Build a synthetic session with `n` size-mismatch recompiles spread across 4 dynamic axes
/// (`x[0..4]`), so clustering does real grouping + per-event guard parsing + value-transition
/// dedup. Constructed via JSON (the real deserialize path) rather than a struct literal —
/// `Session` has many fields and no `Default`, and this keeps the shape honest.
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

fn bench_analyze(c: &mut Criterion) {
    let mut group = c.benchmark_group("recompile_analyze");
    for &n in &[100usize, 1_000, 10_000] {
        let artifact = synthetic_artifact(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &artifact, |b, a| {
            b.iter(|| analyze(a).expect("analyze succeeds"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_analyze);
criterion_main!(benches);
