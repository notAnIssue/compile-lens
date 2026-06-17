//! Integration tests for `cl session report` — the Phase 7 hero report (Tool 7).
//!
//! Spawns the built binary against the committed examples: reads a `.cls.json`, renders the
//! self-contained HTML via `cls-report`, and writes it to `--output` or stdout. The artifact goes
//! through the shared version-gated loader, so a foreign schema version is the same `CLS-E0003`.

use std::io::Write;
use std::process::{Command, Stdio};

fn cl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cl"))
}

fn example(name: &str) -> String {
    format!(
        "{}/../../schema/examples/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    )
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
}

#[test]
fn report_runs_the_fusion_analyzer_for_the_crown_jewel_section() {
    // The fixture carries the GEMM-Residual-RMSNorm-GEMM graphs but **no** stored
    // `fusion_opportunities` — so a populated fusion section proves `cl session report` ran the
    // Tool 6 analyzer (it does not just render stored data). Regression guard for the gap where the
    // crown-jewel section silently rendered "none" for a freshly captured session.
    let out = cl()
        .arg("session")
        .arg("report")
        .arg(fixture("fusion_blocks.cls.json"))
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let html = String::from_utf8_lossy(&out.stdout);
    assert!(html.contains("Fusion opportunities"));
    assert!(
        !html.contains("No algebraic fusion opportunities found"),
        "the analyzer found 3 opportunities in this fixture; the section must not be empty"
    );
    assert!(
        html.contains("epilogue_kit.ops.fused_residual_rms_pattern"),
        "the suggested-kernel column should render from the analyzed opportunities"
    );
}

#[test]
fn report_to_stdout_is_a_self_contained_html_document() {
    let out = cl()
        .arg("session")
        .arg("report")
        .arg(example("full.cls.json"))
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let html = String::from_utf8_lossy(&out.stdout);
    assert!(
        html.starts_with("<!DOCTYPE html>"),
        "stdout must be the HTML document"
    );
    assert!(html.contains("Session metadata"));
    assert!(html.contains("Recompile summary"));
    assert!(!html.contains("https://"), "offline: no external URLs");
}

#[test]
fn report_writes_to_an_output_file() {
    let dst = std::env::temp_dir().join(format!("cls_report_{}.html", std::process::id()));
    let _ = std::fs::remove_file(&dst);

    let out = cl()
        .arg("session")
        .arg("report")
        .arg(example("full.cls.json"))
        .arg("--output")
        .arg(&dst)
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let html = std::fs::read_to_string(&dst).expect("html written");
    let _ = std::fs::remove_file(&dst);
    assert!(html.contains("<!DOCTYPE html>"));
}

#[test]
fn foreign_version_artifact_is_e0003_exit_five() {
    let path = std::env::temp_dir().join(format!("cls_report_bad_{}.cls.json", std::process::id()));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(
            br#"{"schema_version":"9.9.9","session":{"id":"x","timestamp":"t",
"torch_version":"2.6.0","redaction_policy":"default-strict"}}"#,
        )
        .unwrap();

    let out = cl()
        .arg("session")
        .arg("report")
        .arg(&path)
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
}
