//! Registry-level tests for `ClsError`.
//!
//! Guards the public contract D8 requires: every variant has a stable `CLS-EXXXX` code,
//! a non-empty hint, a doc URL, and (where applicable) a cause chain via `#[source]`.
//! Also pins the inherent `code()` and the miette `Diagnostic::code` derive to agree.

use cls_errors::ClsError;
use miette::Diagnostic;

/// Constructs every variant once so tests can iterate the whole enum at once.
fn all_variants() -> Vec<ClsError> {
    vec![
        ClsError::IoError {
            path: "/tmp/x".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        },
        ClsError::SchemaParseError {
            source: serde_json::from_str::<serde_json::Value>("{").unwrap_err(),
        },
        ClsError::SchemaVersionMismatch {
            found: "0.4.0".into(),
            expected: "0.5.0".into(),
        },
        ClsError::InvalidCliArgs {
            detail: "missing --base".into(),
        },
        ClsError::CollectorFailure {
            collector: "recompile".into(),
            source: "boom".into(),
        },
        ClsError::AnalyzerInternalError {
            analyzer: "diff".into(),
            detail: "ICE".into(),
        },
        ClsError::MigrationRequired {
            from: "0.4.0".into(),
            to: "0.5.0".into(),
        },
        ClsError::SensitivePathDetected {
            path: "~/.ssh/id_rsa".into(),
        },
        ClsError::RedactionPolicyDemotionRefused {
            from: "default-strict".into(),
            to: "internal".into(),
        },
        ClsError::WorkingDirectoryAllowlistViolation {
            requested: "/etc/passwd".into(),
        },
        ClsError::NotYetImplemented {
            surface: "cl collect --from-tlparse".into(),
            tracking: "Phase 1 PR plan row #6".into(),
        },
    ]
}

const EXPECTED_CODES: &[&str] = &[
    "CLS-E0001",
    "CLS-E0002",
    "CLS-E0003",
    "CLS-E0004",
    "CLS-E0005",
    "CLS-E0006",
    "CLS-E0007",
    "CLS-E0008",
    "CLS-E0009",
    "CLS-E0010",
    "CLS-E0011",
];

/// Exit-code mapping each variant must produce; aligned with the
/// `EXPECTED_CODES` slice above so a drift between code and exit code surfaces
/// here. `0` is reserved for success and `2` is reserved for clap arg-parse
/// failure (which exits the process before any `ClsError` is constructed).
const EXPECTED_EXIT_CODES: &[i32] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

// ── test_each_variant_has_stable_code ───────────────────────────────────────────────────
#[test]
fn test_each_variant_has_stable_code() {
    let errs = all_variants();
    assert_eq!(
        errs.len(),
        EXPECTED_CODES.len(),
        "must construct every variant"
    );
    for (err, expected) in errs.iter().zip(EXPECTED_CODES.iter()) {
        assert_eq!(err.code(), *expected, "inherent code mismatch");
        // The miette-derive code must agree with the inherent match — catches drift.
        let derived = Diagnostic::code(err)
            .expect("derive must produce a code")
            .to_string();
        assert_eq!(derived, *expected, "miette-derive code mismatch");
    }
}

// ── test_each_variant_has_distinct_exit_code ───────────────────────────────────────────
#[test]
fn test_each_variant_has_distinct_exit_code() {
    let errs = all_variants();
    assert_eq!(
        errs.len(),
        EXPECTED_EXIT_CODES.len(),
        "EXPECTED_EXIT_CODES must mirror EXPECTED_CODES",
    );
    for (err, expected) in errs.iter().zip(EXPECTED_EXIT_CODES.iter()) {
        assert_eq!(
            err.exit_code(),
            *expected,
            "{} exit code mismatch",
            err.code()
        );
    }

    // `0` is success; `2` belongs to clap. No runtime variant may collide.
    for err in &errs {
        let code = err.exit_code();
        assert!(
            code != 0 && code != 2,
            "{} returned reserved exit code",
            err.code()
        );
    }
}

// ── test_each_variant_renders_code_and_help ────────────────────────────────────────────
#[test]
fn test_each_variant_renders_code_and_help() {
    // Every variant supplies the two always-on D8 parts we currently commit to: stable
    // code + actionable help. Cause-chain (#[source]) is variant-dependent and covered by
    // `test_cause_chain_through_source`; doc-URL rendering is deferred (no docs site yet,
    // see ADR-022 + lib.rs crate docs).
    for err in all_variants() {
        let code = err.code();
        assert!(Diagnostic::code(&err).is_some(), "{code} missing code");
        let help = Diagnostic::help(&err)
            .unwrap_or_else(|| panic!("{code} missing help"))
            .to_string();
        assert!(!help.is_empty(), "{code} help is empty");
        // doc URL is intentionally absent until a doc site exists.
        assert!(
            Diagnostic::url(&err).is_none(),
            "{code} unexpectedly carries a URL"
        );
    }
}

// ── test_cls_e0001_io_error ────────────────────────────────────────────────────────────
#[test]
fn test_cls_e0001_io_error() {
    let err = ClsError::IoError {
        path: "/nonexistent".into(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
    };
    assert_eq!(err.code(), "CLS-E0001");
    let display = format!("{err}");
    assert!(
        display.contains("/nonexistent"),
        "display omits path: {display}"
    );
}

// ── test_cls_e0008_sensitive_path_refuses_record ───────────────────────────────────────
#[test]
fn test_cls_e0008_sensitive_path_refuses_record() {
    let err = ClsError::SensitivePathDetected {
        path: "~/.ssh/id_rsa".into(),
    };
    assert_eq!(err.code(), "CLS-E0008");
    let help = Diagnostic::help(&err).unwrap().to_string();
    assert!(
        help.contains(".ssh"),
        "help should mention the refused dirs: {help}"
    );
}

// ── test_cls_e0009_demotion_refused ────────────────────────────────────────────────────
#[test]
fn test_cls_e0009_demotion_refused() {
    let err = ClsError::RedactionPolicyDemotionRefused {
        from: "default-strict".into(),
        to: "internal".into(),
    };
    assert_eq!(err.code(), "CLS-E0009");
    let msg = format!("{err}");
    assert!(
        msg.contains("default-strict") && msg.contains("internal"),
        "message should name both levels: {msg}"
    );
}

// ── test_cause_chain_through_source ────────────────────────────────────────────────────
#[test]
fn test_cause_chain_through_source() {
    // wrap-source variants preserve the underlying error so callers can walk the chain.
    let io_err = ClsError::IoError {
        path: "/x".into(),
        source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
    };
    let cause = std::error::Error::source(&io_err).expect("IoError must expose a cause");
    assert!(cause.to_string().contains("denied"), "cause lost: {cause}");

    // pure variants (no #[source]) return None — call sites won't see a phantom chain.
    let pure = ClsError::InvalidCliArgs { detail: "x".into() };
    assert!(std::error::Error::source(&pure).is_none());
}
