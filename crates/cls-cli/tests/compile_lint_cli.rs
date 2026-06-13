//! Integration tests for the wired `cl compile-lint` subcommand (Tool 4, ADR-032/035).
//!
//! These spawn the built `cl` binary. `compile-lint` is the analysis half: it loads a `.cls.json`
//! whose `lint_findings[]` were filled by the Python source scan, joins each candidate with the
//! correctness database, escalates severity by the user's torch version, and renders. It exits 1
//! when a `high` finding survives so it can gate CI — distinct from the exit code 6 a bad argument
//! produces.
//!
//! Fixtures: `tests/fixtures/unfixed.cls.json` pins torch 2.4.0, below the database's `fixed_in`
//! 2.5, so its one candidate escalates to `high`. `schema/examples/full.cls.json` pins torch 2.6.0,
//! at/above the fix, so the same candidate downgrades to `info`.

use std::process::Command;

fn cl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cl"))
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn example(name: &str) -> String {
    format!(
        "{}/../../schema/examples/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    )
}

#[test]
fn high_finding_renders_evidence_and_gates_with_exit_1() {
    let out = cl()
        .arg("compile-lint")
        .arg(fixture("unfixed.cls.json"))
        .arg("--db")
        .arg(fixture("correctness_db.toml"))
        .output()
        .expect("spawn cl");

    // torch 2.4 < fixed_in 2.5 -> the bug bites -> `high` -> exit 1 (the CI gate).
    assert_eq!(out.status.code(), Some(1), "expected the high-finding gate");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("high"), "missing severity:\n{stdout}");
    assert!(
        stdout.contains("in_place_op_on_alias"),
        "missing pattern:\n{stdout}"
    );
    assert!(stdout.contains("issues/114302"), "missing issue:\n{stdout}");
    assert!(
        stdout.contains("model.py:47"),
        "missing location:\n{stdout}"
    );
}

#[test]
fn fixed_for_user_torch_downgrades_to_info_and_passes() {
    let out = cl()
        .arg("compile-lint")
        .arg(example("full.cls.json"))
        .arg("--db")
        .arg(fixture("correctness_db.toml"))
        .output()
        .expect("spawn cl");

    // torch 2.6 >= fixed_in 2.5 -> already fixed -> `info`, no `high` -> success.
    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("info"), "expected downgrade:\n{stdout}");
}

#[test]
fn empty_db_drops_every_candidate() {
    // No `--db` -> empty database -> no candidate has a cited issue -> nothing is reported
    // (lint-not-oracle). The command succeeds: an absent database is not an error.
    let out = cl()
        .arg("compile-lint")
        .arg(fixture("unfixed.cls.json"))
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("No lint findings"), "{stdout}");
}

#[test]
fn sarif_is_valid_and_marks_the_high_finding_as_error() {
    let out = cl()
        .arg("compile-lint")
        .arg(fixture("unfixed.cls.json"))
        .arg("--db")
        .arg(fixture("correctness_db.toml"))
        .arg("--format")
        .arg("sarif")
        .output()
        .expect("spawn cl");

    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    assert_eq!(v["version"], "2.1.0");
    assert_eq!(v["runs"][0]["tool"]["driver"]["name"], "compile-lens");
    // high -> SARIF `error`, the level GitHub Code Scanning surfaces as a failure.
    assert_eq!(v["runs"][0]["results"][0]["level"], "error");
}

#[test]
fn json_format_round_trips() {
    let out = cl()
        .arg("compile-lint")
        .arg(fixture("unfixed.cls.json"))
        .arg("--db")
        .arg(fixture("correctness_db.toml"))
        .arg("--format")
        .arg("json")
        .output()
        .expect("spawn cl");

    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert_eq!(v["findings"][0]["severity"], "high");
    assert_eq!(v["findings"][0]["pattern_category"], "in_place_op_on_alias");
}

#[test]
fn min_severity_high_filters_out_the_info_finding() {
    // full.cls.json's only finding downgrades to `info`; `--min-severity high` drops it, leaving
    // nothing — and with no `high` left the command passes.
    let out = cl()
        .arg("compile-lint")
        .arg(example("full.cls.json"))
        .arg("--db")
        .arg(fixture("correctness_db.toml"))
        .arg("--min-severity")
        .arg("high")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("No lint findings"), "{stdout}");
}

#[test]
fn unrecognized_min_severity_is_rejected() {
    let out = cl()
        .arg("compile-lint")
        .arg(fixture("unfixed.cls.json"))
        .arg("--db")
        .arg(fixture("correctness_db.toml"))
        .arg("--min-severity")
        .arg("nope")
        .output()
        .expect("spawn cl");

    // A bad argument is CLS-E0004 -> exit 6, never the value `1` the CI gate reserves.
    assert_eq!(out.status.code(), Some(6), "expected arg-validation exit");
}
