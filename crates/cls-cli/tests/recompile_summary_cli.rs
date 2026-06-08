//! Integration tests for the wired `cl recompile-summary` subcommand.
//!
//! These spawn the built `cl` binary against the committed example artifacts in
//! `schema/examples/`, exercising the full path the Python front-end drives: load the
//! `.cls.json` → run Tool 1 (analyze, or diff against `--baseline`) → render the chosen
//! `--format` → print to stdout. Spawning the process (rather than calling `run()`) is what
//! lets us assert the exit-code contract and that machine-readable JSON lands on stdout.

use std::io::Write;
use std::process::{Command, Stdio};

fn cl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cl"))
}

/// A committed example artifact, addressed relative to this crate's manifest dir
/// (`crates/cls-cli`) so the test does not depend on the process working directory.
fn example(name: &str) -> String {
    format!(
        "{}/../../schema/examples/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    )
}

/// `full.cls.json` has one recompilation, so its summary is non-empty.
#[test]
fn summary_json_is_valid_and_reports_the_recompile() {
    let out = cl()
        .arg("recompile-summary")
        .arg(example("full.cls.json"))
        .arg("--format")
        .arg("json")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    assert_eq!(value["total_recompilations"], 1);
}

#[test]
fn summary_markdown_has_heading() {
    let out = cl()
        .arg("recompile-summary")
        .arg(example("full.cls.json"))
        .arg("--format")
        .arg("markdown")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("## Recompile Summary"),
        "markdown missing heading:\n{stdout}"
    );
}

/// `markdown` is the documented default, so omitting `--format` matches the explicit form.
#[test]
fn summary_defaults_to_markdown() {
    let out = cl()
        .arg("recompile-summary")
        .arg(example("full.cls.json"))
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("## Recompile Summary"), "stdout:\n{stdout}");
}

/// `minimal.cls.json` (baseline, 0 recompiles) vs `full.cls.json` (head, 1 recompile) →
/// the diff should report the recompile as *added*.
#[test]
fn baseline_diff_reports_added_cluster() {
    let out = cl()
        .arg("recompile-summary")
        .arg(example("full.cls.json"))
        .arg("--baseline")
        .arg(example("minimal.cls.json"))
        .arg("--format")
        .arg("markdown")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("## Recompile Diff (vs baseline)"),
        "diff missing heading:\n{stdout}"
    );
    assert!(
        stdout.contains("Added: 1"),
        "diff should show 1 added:\n{stdout}"
    );
}

/// A file that is not valid JSON → typed `SchemaParseError` (CLS-E0002) → exit 4.
#[test]
fn malformed_artifact_exits_four() {
    let path = std::env::temp_dir().join(format!("cls_bad_{}.cls.json", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(b"this is not json {{{")
            .expect("write temp file");
    }

    let out = cl()
        .arg("recompile-summary")
        .arg(&path)
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    let _ = std::fs::remove_file(&path);

    assert_eq!(
        out.status.code(),
        Some(4),
        "expected exit 4 (CLS-E0002), got {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("CLS-E0002"),
        "diagnostic missing CLS-E0002 marker"
    );
}
