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
fn diff_runs_on_fixture_pairs_and_classifies_changes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/diff");
    let run = |scenario: &str| -> (Option<i32>, String) {
        let dir = root.join(scenario);
        let out = cl()
            .arg("diff")
            .arg("--base")
            .arg(dir.join("base.cls.json"))
            .arg("--head")
            .arg(dir.join("head.cls.json"))
            .arg("--format")
            .arg("markdown")
            .output()
            .expect("spawn cl");
        (
            out.status.code(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    };

    // Runs cleanly (exit 0) on several real fixture pairs.
    for scenario in [
        "add_node",
        "remove_node",
        "operand_swap_sub",
        "commutative_add",
    ] {
        let (code, stdout) = run(scenario);
        assert_eq!(code, Some(0), "{scenario}: cl diff should exit 0");
        assert!(
            stdout.contains("## Compile Diff"),
            "{scenario}: markdown header missing:\n{stdout}"
        );
        // cl diff also carries the cache-stability diff section (clean on graph-only fixtures).
        assert!(
            stdout.contains("## Cache Stability (diff)"),
            "{scenario}: cache-stability section missing:\n{stdout}"
        );
    }

    // And classifies each correctly, through the CLI end to end.
    let added = run("add_node").1;
    assert!(
        added.contains("Added") && added.contains("n3"),
        "add_node:\n{added}"
    );
    let swapped = run("operand_swap_sub").1;
    assert!(
        swapped.contains("Modified") && swapped.contains("n2"),
        "operand_swap_sub:\n{swapped}"
    );
    assert!(
        run("commutative_add").1.contains("No structural changes"),
        "commutative_add should be silent"
    );
}

#[test]
fn cache_stability_detects_listing2_fixture() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cache_stability/listing2.cls.json");
    let out = cl()
        .arg("cache-stability")
        .arg(&fixture)
        .arg("--format")
        .arg("markdown")
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(0),
        "cl cache-stability should exit 0; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("## Cache Stability"),
        "header missing:\n{stdout}"
    );
    assert!(
        stdout.contains("iteration 1") && stdout.contains("step_counter"),
        "Listing 2 finding missing:\n{stdout}"
    );
}

#[test]
fn cache_stability_nonexistent_session_exits_three() {
    let out = cl()
        .arg("cache-stability")
        .arg("/nonexistent/session.cls.json")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(3),
        "missing file → CLS-E0001 exit 3"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("CLS-E0001"),
        "diagnostic missing CLS-E0001 marker"
    );
}

/// False-positive baseline: a *normal* stateful session — state drifts, but the cache correctly
/// invalidates (recompiles) so the output moves — must produce no finding.
#[test]
fn cache_stability_no_false_positive_on_normal_stateful() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cache_stability/normal_stateful.cls.json");
    let out = cl()
        .arg("cache-stability")
        .arg(&fixture)
        .output()
        .expect("spawn cl");

    assert_eq!(out.status.code(), Some(0), "should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("No cache-stability anomalies detected"),
        "normal stateful session must not false-positive:\n{stdout}"
    );
}
