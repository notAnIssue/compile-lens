//! Integration tests for `load_artifact` — the shared baseline-pairing loader (ADR-033 §5).
//!
//! `load_artifact` is the one seam every analyzer goes through to read a *parsed* `.cls.json` off
//! disk: it version-gates first (turning a foreign version into the actionable `CLS-E0003`) and
//! then deserializes into a typed `ClsArtifact`. These tests cover its three outcomes — a clean
//! load, a version mismatch, and a well-versioned but unparseable body.

use std::path::PathBuf;

use cls_errors::ClsError;
use cls_schema_migrate::{load_artifact, CURRENT_VERSION};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("migrate")
        .join(name)
}

fn tempfile_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("cls-load-artifact-{}-{}", std::process::id(), name))
}

// ── success ──────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_load_current_artifact_parses() {
    let artifact = load_artifact(&fixture("v050.cls.json")).expect("current artifact loads");
    assert_eq!(artifact.schema_version, CURRENT_VERSION);
}

// ── version mismatch -> CLS-E0003 ──────────────────────────────────────────────────────────────
#[test]
fn test_foreign_version_is_e0003_not_a_serde_error() {
    // A future schema version is refused at the version gate (pre-V1: the user re-collects). The
    // point of the gate is that this is an *actionable* CLS-E0003, not a noisy serde error about
    // whatever field the foreign artifact happens to differ on.
    let tmp = tempfile_path("foreign.cls.json");
    std::fs::write(
        &tmp,
        r#"{"schema_version":"9.9.9","session":{"id":"x","timestamp":"t",
"torch_version":"2.6.0","redaction_policy":"default-strict"}}"#,
    )
    .unwrap();

    let err = load_artifact(&tmp).expect_err("foreign version must error");
    let _ = std::fs::remove_file(&tmp);

    match err {
        ClsError::SchemaVersionMismatch { found, expected } => {
            assert_eq!(found, "9.9.9");
            assert_eq!(expected, CURRENT_VERSION);
        }
        other => panic!("expected SchemaVersionMismatch (CLS-E0003), got {other:?}"),
    }
}

// ── current version, but the body won't deserialize -> parse error ─────────────────────────────
#[test]
fn test_current_version_but_invalid_body_is_parse_error() {
    // schema_version is current, so the version gate passes; but the body is missing the required
    // `session`, so the *typed* parse fails. This is the branch the version gate cannot catch —
    // detect_schema_version only inspects `schema_version`, not the rest of the document.
    let tmp = tempfile_path("invalid_body.cls.json");
    std::fs::write(&tmp, r#"{"schema_version":"0.5.0"}"#).unwrap();

    let err = load_artifact(&tmp).expect_err("missing required fields must error");
    let _ = std::fs::remove_file(&tmp);

    assert!(
        matches!(err, ClsError::SchemaParseError { .. }),
        "expected SchemaParseError, got {err:?}"
    );
}
