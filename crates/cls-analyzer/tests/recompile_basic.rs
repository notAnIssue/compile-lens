//! Tests for the Tool 1 analyzer skeleton.
//!
//! The implementation is intentionally absent — the upcoming Phase 1 PRs
//! fill it in. These tests pin the **surface** so the typed errors,
//! default-empty behaviour, and signature contract don't drift while the
//! algorithm work is in flight.

use cls_analyzer::recompile;
use cls_schema::{RedactionPolicy, Session};

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
fn analyze_empty_session_returns_default_findings() {
    // Empty session → default findings. This is the call-site smoke test
    // that proves the surface is wired: callers can `analyze(&session)`
    // and get a `RecompileFindings` without hitting the not-yet-implemented
    // branch.
    let session = minimal_session();
    let findings = recompile::analyze(&session).expect("empty session should not error");
    assert_eq!(findings.total_recompilations, 0);
    assert!(findings.guard_categories.is_empty());
    assert!(findings.top_suggestions.is_empty());
}

#[test]
fn analyze_session_with_recompiles_is_not_yet_implemented() {
    // Once `iteration_count > 0` we're in the actual analyzer territory,
    // which has not landed. The skeleton returns `NotYetImplemented` with
    // the surface and tracking pointer named.
    let mut session = minimal_session();
    session.iteration_count = Some(3);

    let err = recompile::analyze(&session).expect_err("nonempty session should not be ok");
    assert_eq!(err.code(), "CLS-E0011");
    assert_eq!(err.exit_code(), 13);
}
