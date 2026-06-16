//! Integration tests for the wired `cl kernel-roofline` subcommand (Tool 5) at the binary level.
//!
//! `tests/integration/test_tool5_e2e.py` drives the same command via `cargo run`; these spawn the
//! built binary directly to assert what the Python e2e does not: the exit-code contract for an
//! unknown `--gpu` (6 = invalid CLI args), the `--list` short-circuit, and the `--kernel` id filter.

use std::process::{Command, Stdio};

fn cl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cl"))
}

/// The canonical Tool 5 fixture: four kernel configs with features + measurements.
fn fixture() -> String {
    format!(
        "{}/../../tests/fixtures/roofline/sample.cls.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn unknown_gpu_is_invalid_args_exit_six() {
    let out = cl()
        .arg("kernel-roofline")
        .arg(fixture())
        .arg("--gpu")
        .arg("NOT-A-GPU")
        .stderr(Stdio::piped())
        .output()
        .expect("spawn cl");

    assert_eq!(
        out.status.code(),
        Some(6),
        "expected exit 6 (InvalidCliArgs); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .to_lowercase()
            .contains("gpu"),
        "diagnostic should name the unknown GPU"
    );
}

#[test]
fn list_enumerates_kernels_and_exits_zero() {
    let out = cl()
        .arg("kernel-roofline")
        .arg(fixture())
        .arg("--list")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let ids: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').next().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "cfg_bs256_r32",
            "cfg_bs64_r32",
            "cfg_bs32_r32",
            "cfg_bs256_r64"
        ]
    );
}

#[test]
fn kernel_filter_narrows_the_json_report() {
    let out = cl()
        .arg("kernel-roofline")
        .arg(fixture())
        .arg("--kernel")
        .arg("bs64")
        .arg("--format")
        .arg("json")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let value: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).expect("stdout is JSON");
    let ids: Vec<&str> = value["predictions"]
        .as_array()
        .expect("predictions array")
        .iter()
        .map(|p| p["kernel_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["cfg_bs64_r32"], "only the bs64 config should remain");
}
