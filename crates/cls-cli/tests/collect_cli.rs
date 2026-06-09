//! Integration tests for the `cl collect` and `cl recompile-summary`
//! subcommand surface.
//!
//! These run the built `cl` binary as a subprocess and assert on the exit
//! code and rendered diagnostic. Spawning the binary (rather than calling
//! `run()` directly) is intentional: the exit-code contract is a property
//! of the process, not the function, and clap's `process::exit` path on
//! arg-parse failure cannot be observed any other way.

use std::process::{Command, Stdio};

/// Path to the built `cl` binary the runner produces for this crate.
fn cl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cl"))
}

#[test]
fn collect_help_lists_mode_flags() {
    let out = cl()
        .arg("collect")
        .arg("--help")
        .output()
        .expect("spawn cl");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit: {:?}", out.status);
    assert!(help.contains("--from-logs"), "help missing --from-logs");
    assert!(
        help.contains("--from-tlparse"),
        "help missing --from-tlparse"
    );
    assert!(
        help.contains("--from-dynamo-explain"),
        "help missing --from-dynamo-explain"
    );
    assert!(help.contains("--output"), "help missing --output");
    assert!(help.contains("--redaction"), "help missing --redaction");
}

#[test]
fn recompile_summary_help_lists_format_flag() {
    let out = cl()
        .arg("recompile-summary")
        .arg("--help")
        .output()
        .expect("spawn cl");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit: {:?}", out.status);
    assert!(help.contains("--format"), "help missing --format");
}

#[test]
fn collect_from_logs_nonexistent_path_exits_three() {
    // Mode A with a path that does not exist → CLS-E0001 (IoError) → exit 3.
    // The PR plan spec calls this out as the canonical demonstration that
    // typed-error → exit-code mapping is wired.
    let out = cl()
        .arg("collect")
        .arg("--from-logs")
        .arg("/nonexistent/path/recompiles.log")
        .arg("--output")
        .arg("/tmp/never_written.json")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(3),
        "expected exit 3 (CLS-E0001), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let rendered = String::from_utf8_lossy(&out.stderr);
    assert!(
        rendered.contains("CLS-E0001"),
        "rendered diagnostic missing CLS-E0001 marker:\n{rendered}"
    );
}

#[test]
fn collect_no_mode_exits_six_with_help() {
    // Neither `--from-logs` nor `--from-tlparse` nor `--from-dynamo-explain`
    // supplied → InvalidCliArgs (CLS-E0004) → exit 6. The help text in the
    // rendered diagnostic tells the user which flags are options.
    let out = cl()
        .arg("collect")
        .arg("--output")
        .arg("/tmp/never_written.json")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(6),
        "expected exit 6 (CLS-E0004), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let rendered = String::from_utf8_lossy(&out.stderr);
    assert!(
        rendered.contains("CLS-E0004"),
        "rendered diagnostic missing CLS-E0004 marker:\n{rendered}"
    );
}

#[test]
fn collect_from_dynamo_explain_exits_thirteen() {
    // `--from-dynamo-explain` takes no path and parses cleanly → falls
    // through to `NotYetImplemented` (CLS-E0011) → exit 13.
    let out = cl()
        .arg("collect")
        .arg("--from-dynamo-explain")
        .arg("--output")
        .arg("/tmp/never_written.json")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(13),
        "expected exit 13 (CLS-E0011), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let rendered = String::from_utf8_lossy(&out.stderr);
    assert!(
        rendered.contains("CLS-E0011"),
        "rendered diagnostic missing CLS-E0011 marker:\n{rendered}"
    );
    assert!(
        rendered.contains("cl collect --from-dynamo-explain"),
        "diagnostic should name the unimplemented surface:\n{rendered}"
    );
}

#[test]
fn recompile_summary_nonexistent_session_exits_three() {
    let out = cl()
        .arg("recompile-summary")
        .arg("/nonexistent/session.cls.json")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(3),
        "expected exit 3 (CLS-E0001), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn diff_help_lists_base_and_head() {
    let out = cl().arg("diff").arg("--help").output().expect("spawn cl");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit: {:?}", out.status);
    assert!(help.contains("--base"), "help missing --base");
    assert!(help.contains("--head"), "help missing --head");
    assert!(help.contains("--format"), "help missing --format");
}

#[test]
fn diff_nonexistent_base_exits_three() {
    // `--base` points at a missing file → CLS-E0001 (IoError) → exit 3, before the
    // not-yet-implemented branch. (Both inputs are validated; base is checked first.)
    let out = cl()
        .arg("diff")
        .arg("--base")
        .arg("/nonexistent/base.cls.json")
        .arg("--head")
        .arg("/nonexistent/head.cls.json")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(3),
        "expected exit 3 (CLS-E0001), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("CLS-E0001"),
        "diagnostic missing CLS-E0001 marker"
    );
}

#[test]
fn diff_with_existing_inputs_exits_not_yet_implemented() {
    // Both inputs reachable → falls through to NotYetImplemented (CLS-E0011) → exit 13.
    let base = std::env::temp_dir().join(format!("cls_diff_base_{}.cls.json", std::process::id()));
    let head = std::env::temp_dir().join(format!("cls_diff_head_{}.cls.json", std::process::id()));
    std::fs::write(&base, b"{}").expect("write base");
    std::fs::write(&head, b"{}").expect("write head");

    let out = cl()
        .arg("diff")
        .arg("--base")
        .arg(&base)
        .arg("--head")
        .arg(&head)
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    let _ = std::fs::remove_file(&base);
    let _ = std::fs::remove_file(&head);

    assert_eq!(
        out.status.code(),
        Some(13),
        "expected exit 13 (CLS-E0011), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cl diff"),
        "diagnostic should name the unimplemented surface"
    );
}
