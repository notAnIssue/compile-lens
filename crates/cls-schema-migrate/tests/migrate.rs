//! Integration tests for `cls-schema-migrate`.
//!
//! Pre-V1 the contract is "detect and refuse": detect_schema_version recognizes the
//! current version, rejects anything else with `CLS-E0003`, and `migrate_to_current`
//! byte-copies when the input is already at the current version. The migration ladder
//! activates at V1 (see lib.rs).

use std::path::PathBuf;

use cls_errors::ClsError;
use cls_schema_migrate::{detect_schema_version, migrate_to_current, CURRENT_VERSION};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("migrate")
        .join(name)
}

fn tempfile_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("cls-schema-migrate-{}-{}", std::process::id(), name))
}

// ── test_detect_current_schema ─────────────────────────────────────────────────────────
#[test]
fn test_detect_current_schema() {
    let version = detect_schema_version(&fixture("v050.cls.json")).expect("detect ok");
    assert_eq!(version, CURRENT_VERSION);
    // Sanity: this crate's notion of "current" must match the schema crate's constant.
    assert_eq!(CURRENT_VERSION, "0.5.0");
}

// ── test_detect_unknown_version_returns_error ──────────────────────────────────────────
#[test]
fn test_detect_unknown_version_returns_error() {
    let tmp = tempfile_path("bogus.cls.json");
    std::fs::write(
        &tmp,
        r#"{"schema_version":"9.9.9","session":{"id":"x","timestamp":"t",
"torch_version":"2.6.0","redaction_policy":"default-strict"}}"#,
    )
    .unwrap();

    let err = detect_schema_version(&tmp).expect_err("unknown version must error");
    let _ = std::fs::remove_file(&tmp);

    match err {
        ClsError::SchemaVersionMismatch { found, expected } => {
            assert_eq!(found, "9.9.9");
            assert_eq!(expected, CURRENT_VERSION);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

// ── test_detect_missing_version_returns_error ──────────────────────────────────────────
#[test]
fn test_detect_missing_version_returns_error() {
    // No `schema_version` field at all -> `found = "(missing)"` (DR-4: reuse the existing
    // variant rather than introducing a parallel `SchemaVersionMissing` for one case).
    let tmp = tempfile_path("nofield.cls.json");
    std::fs::write(&tmp, r#"{"session":{"id":"x"}}"#).unwrap();

    let err = detect_schema_version(&tmp).expect_err("missing field must error");
    let _ = std::fs::remove_file(&tmp);

    match err {
        ClsError::SchemaVersionMismatch { found, .. } => {
            assert_eq!(found, "(missing)");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

// ── test_byte_copy_when_input_is_current ───────────────────────────────────────────────
#[test]
fn test_byte_copy_when_input_is_current() {
    let input = fixture("v050.cls.json");
    let output = tempfile_path("byte_copy_out.cls.json");

    migrate_to_current(&input, &output).expect("matching-current case is byte-copy");

    let input_bytes = std::fs::read(&input).unwrap();
    let output_bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(&output);

    // D10: same input -> byte-identical output (verifiable by `cmp`). A post-V1
    // per-version migration legitimately reformats; this pre-V1 case is byte-copy by design.
    assert_eq!(input_bytes, output_bytes);
}

// ── test_migrate_to_current_refuses_unknown_before_writing ─────────────────────────────
#[test]
fn test_migrate_to_current_refuses_unknown_before_writing() {
    let bogus = tempfile_path("dispatch_bogus.cls.json");
    std::fs::write(
        &bogus,
        r#"{"schema_version":"9.9.9","session":{"id":"x","timestamp":"t",
"torch_version":"2.6.0","redaction_policy":"default-strict"}}"#,
    )
    .unwrap();

    let out = tempfile_path("dispatch_bogus_out.cls.json");
    // Ensure no stale file from a previous run before we assert it stays absent.
    let _ = std::fs::remove_file(&out);

    let err = migrate_to_current(&bogus, &out).expect_err("unknown version must error");
    let _ = std::fs::remove_file(&bogus);

    assert!(matches!(err, ClsError::SchemaVersionMismatch { .. }));
    // And the output file must NOT have been written — refusal happens before any IO.
    assert!(!out.exists(), "output should not be created on detection failure");
}
