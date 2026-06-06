//! Surface tests for the Tool 1 analyzer entry point.
//!
//! Pins the `analyze(&ClsArtifact)` signature and the empty-artifact contract; the clustering
//! behaviour itself is covered in `recompile_cluster.rs`.

use cls_analyzer::recompile;
use cls_schema::{ClsArtifact, RedactionPolicy, Session};

fn minimal_session() -> Session {
    Session {
        id: "test-id".into(),
        timestamp: "2026-06-01T00:00:00Z".into(),
        torch_version: "2.12.0".into(),
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

#[test]
fn analyze_empty_artifact_returns_default_findings() {
    // An artifact with no recompiles → zero count, no clusters, no suggestions, no error.
    let artifact = ClsArtifact::new(minimal_session());
    let findings = recompile::analyze(&artifact).expect("empty artifact should not error");
    assert_eq!(findings.total_recompilations, 0);
    assert!(findings.guard_categories.is_empty());
    assert!(findings.top_suggestions.is_empty());
}
