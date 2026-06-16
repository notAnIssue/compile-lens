//! Integration tests for the wired `cl migrate` subcommand.
//!
//! Spawns the built `cl` binary to assert the parts that only exist at the process boundary: the
//! exit-code contract (5 = CLS-E0003 schema-version mismatch, 6 = invalid CLI args) and the two
//! success modes — `--dry-run` (report, no write) and `--output` (pre-V1 byte-copy, D10).

use std::io::Write;
use std::process::{Command, Stdio};

fn cl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cl"))
}

/// A committed example artifact, addressed relative to this crate's manifest dir.
fn example(name: &str) -> String {
    format!(
        "{}/../../schema/examples/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    )
}

fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cls_migrate_{}_{}", std::process::id(), name));
    std::fs::File::create(&path)
        .expect("create temp file")
        .write_all(body.as_bytes())
        .expect("write temp file");
    path
}

#[test]
fn dry_run_on_current_artifact_reports_no_changes() {
    let out = cl()
        .arg("migrate")
        .arg(example("minimal.cls.json"))
        .arg("--dry-run")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("no changes"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("0.5.0"),
        "should name the current schema:\n{stdout}"
    );
}

#[test]
fn output_byte_copies_a_current_artifact() {
    let src = example("minimal.cls.json");
    let dst = std::env::temp_dir().join(format!("cls_migrate_out_{}.cls.json", std::process::id()));
    let _ = std::fs::remove_file(&dst);

    let out = cl()
        .arg("migrate")
        .arg(&src)
        .arg("--output")
        .arg(&dst)
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    // D10: pre-V1 migrate of an already-current artifact is a byte-identical copy.
    let src_bytes = std::fs::read(&src).expect("read src");
    let dst_bytes = std::fs::read(&dst).expect("read dst");
    let _ = std::fs::remove_file(&dst);
    assert_eq!(
        src_bytes, dst_bytes,
        "migrate --output must byte-copy a current artifact"
    );
}

#[test]
fn neither_output_nor_dry_run_is_invalid_args_exit_six() {
    // No mode selected -> typed InvalidCliArgs (CLS-E0006), exit 6 — not a panic or a silent no-op.
    let out = cl()
        .arg("migrate")
        .arg(example("minimal.cls.json"))
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(6),
        "expected exit 6 (InvalidCliArgs); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn unknown_version_is_e0003_exit_five() {
    // A foreign schema version is refused at the version gate: CLS-E0003, exit 5. This is the
    // exit code that had no binary-level coverage before.
    let path = write_temp(
        "bogus.cls.json",
        r#"{"schema_version":"9.9.9","session":{"id":"x","timestamp":"t",
"torch_version":"2.6.0","redaction_policy":"default-strict"}}"#,
    );

    let out = cl()
        .arg("migrate")
        .arg(&path)
        .arg("--dry-run")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    let _ = std::fs::remove_file(&path);
    assert_eq!(
        out.status.code(),
        Some(5),
        "expected exit 5 (CLS-E0003); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("CLS-E0003"),
        "diagnostic missing CLS-E0003 marker"
    );
}
