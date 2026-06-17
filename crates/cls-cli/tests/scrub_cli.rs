//! Integration tests for the wired `cl scrub` subcommand.
//!
//! Spawns the built `cl` binary to assert the parts that only exist at the process boundary: the
//! write/dry-run modes, the demotion-refusal exit-code contract (11 = CLS-E0009), and that HTML
//! input is routed to the not-yet-implemented path (13). `CLS_INSTALL_ID` is pinned so the host
//! hash is deterministic across runs.

use std::io::Write;
use std::process::{Command, Stdio};

fn cl() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_cl"));
    // Deterministic salt so `dh-<hash>` is stable and never touches ~/.compile-lens.
    cmd.env("CLS_INSTALL_ID", "test-fixed-salt");
    cmd
}

/// A confidential artifact that leaks a host, a tokened command, and a kernel source path.
const LEAKY: &str = r#"{
  "schema_version": "0.5.0",
  "session": {
    "id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-06-17T00:00:00Z",
    "torch_version": "2.11.0",
    "redaction_policy": "confidential",
    "host": "ml-prod-07.megacorp.internal",
    "command": "python train.py --hf-token=hf_abcdefghijklmnopqrst"
  },
  "kernels": [
    {"kernel_id": "k0", "name": "fused", "ptx_path": "/tmp/x.ptx", "kernel_source_excerpt": "secret"}
  ]
}"#;

fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cls_scrub_{}_{}", std::process::id(), name));
    std::fs::File::create(&path)
        .expect("create temp file")
        .write_all(body.as_bytes())
        .expect("write temp file");
    path
}

#[test]
fn scrub_in_place_promotes_confidential_to_default_strict() {
    let path = write_temp("leaky.cls.json", LEAKY);
    let out = cl().arg("scrub").arg(&path).output().expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());

    let scrubbed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read back")).expect("valid json");
    let _ = std::fs::remove_file(&path);

    let session = &scrubbed["session"];
    assert_eq!(session["redaction_policy"], "default-strict");
    // Host hashed, not raw.
    assert!(session["host"].as_str().unwrap().starts_with("dh-"));
    assert!(!session["host"].as_str().unwrap().contains("megacorp"));
    // Token scrubbed, flag kept.
    let cmd = session["command"].as_str().unwrap();
    assert!(cmd.contains("--hf-token=<scrubbed>"));
    assert!(!cmd.contains("hf_abcdefghijklmnopqrst"));
    // Kernel IP gone (serialized away because the fields became null).
    let kernel = &scrubbed["kernels"][0];
    assert!(kernel.get("ptx_path").is_none());
    assert!(kernel.get("kernel_source_excerpt").is_none());
}

#[test]
fn scrub_to_public_safe_nulls_host_and_command() {
    let path = write_temp("pub.cls.json", LEAKY);
    let out = cl()
        .arg("scrub")
        .arg(&path)
        .arg("--to")
        .arg("public-safe")
        .output()
        .expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());

    let scrubbed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read back")).expect("valid json");
    let _ = std::fs::remove_file(&path);
    let session = &scrubbed["session"];
    assert_eq!(session["redaction_policy"], "public-safe");
    assert!(session.get("host").is_none(), "public-safe nulls host");
    assert!(
        session.get("command").is_none(),
        "public-safe nulls command"
    );
}

#[test]
fn dry_run_reports_but_does_not_write() {
    let path = write_temp("dry.cls.json", LEAKY);
    let before = std::fs::read(&path).expect("read before");

    let out = cl()
        .arg("scrub")
        .arg(&path)
        .arg("--dry-run")
        .output()
        .expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());

    let after = std::fs::read(&path).expect("read after");
    let _ = std::fs::remove_file(&path);
    assert_eq!(before, after, "--dry-run must not modify the file");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("would scrub"), "stdout:\n{stdout}");
    assert!(stdout.contains("default-strict"), "stdout:\n{stdout}");
}

#[test]
fn demote_is_refused_exit_eleven() {
    // A default-strict artifact asked to become `internal` (less strict) -> CLS-E0009, exit 11.
    let already = LEAKY.replace("\"confidential\"", "\"default-strict\"");
    let path = write_temp("demote.cls.json", &already);

    let out = cl()
        .arg("scrub")
        .arg(&path)
        .arg("--to")
        .arg("internal")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        out.status.code(),
        Some(11),
        "expected exit 11 (CLS-E0009); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("CLS-E0009"),
        "diagnostic missing CLS-E0009 marker"
    );
}

#[test]
fn verify_flags_a_leaky_artifact_exit_one() {
    // A confidential artifact audited against default-strict still leaks -> exit 1, ⚠️ lines.
    let path = write_temp("verify_leaky.cls.json", LEAKY);
    let out = cl()
        .arg("scrub")
        .arg("--verify")
        .arg(&path)
        .arg("--target")
        .arg("default-strict")
        .output()
        .expect("spawn cl");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        out.status.code(),
        Some(1),
        "a leaky artifact must fail verify"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("NOT default-strict"), "stdout:\n{stdout}");
}

#[test]
fn verify_passes_after_scrub_exit_zero() {
    // Scrub first, then audit the result: clean, exit 0, SHARE-SAFE verdict.
    let path = write_temp("verify_clean.cls.json", LEAKY);
    let scrub = cl().arg("scrub").arg(&path).output().expect("spawn cl");
    assert!(scrub.status.success());

    let out = cl()
        .arg("scrub")
        .arg("--verify")
        .arg(&path)
        .output()
        .expect("spawn cl");
    let _ = std::fs::remove_file(&path);

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("SHARE-SAFE"), "stdout:\n{stdout}");
}

#[test]
fn html_input_is_sanitized_in_place() {
    // An HTML report with no CSP and a script tag gets the strict CSP added and the script made
    // inert.
    let evil = "<html><head><title>r</title></head><body><script>steal()</script></body></html>";
    let path = write_temp("report.html", evil);
    let out = cl().arg("scrub").arg(&path).output().expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());

    let scrubbed = std::fs::read_to_string(&path).expect("read back");
    let _ = std::fs::remove_file(&path);
    assert!(scrubbed.contains("default-src 'none'"), "CSP added");
    assert!(
        !scrubbed.contains("<script"),
        "script neutralized: {scrubbed}"
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("CSP added"));
}
