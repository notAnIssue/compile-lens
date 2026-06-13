//! Integration tests for the wired `cl divergence-view` subcommand (ADR-034).
//!
//! These spawn the built `cl` binary against the committed example artifacts in
//! `schema/examples/`. `divergence-view` is view-only: it loads a `.cls.json` and renders the
//! stored `divergences[]` — no analysis, no torch. `full.cls.json` carries one divergence (with a
//! nested attribution); `minimal.cls.json` carries none.

use std::process::Command;

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

#[test]
fn view_json_is_valid_and_reports_the_divergence() {
    let out = cl()
        .arg("divergence-view")
        .arg(example("full.cls.json"))
        .arg("--format")
        .arg("json")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    assert_eq!(
        value[0]["first_divergent_layer"],
        "decoder.layers.3.mlp.fc2"
    );
    assert_eq!(
        value[0]["attribution"]["responsible_passes"][0],
        "post_grad_custom_post_pass"
    );
}

#[test]
fn view_markdown_has_heading_and_layer() {
    let out = cl()
        .arg("divergence-view")
        .arg(example("full.cls.json"))
        .arg("--format")
        .arg("markdown")
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("## Divergence"),
        "missing heading:\n{stdout}"
    );
    assert!(
        stdout.contains("decoder.layers.3.mlp.fc2"),
        "missing layer:\n{stdout}"
    );
}

/// `markdown` is the documented default, so omitting `--format` matches the explicit form.
#[test]
fn view_defaults_to_markdown() {
    let out = cl()
        .arg("divergence-view")
        .arg(example("full.cls.json"))
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("## Divergence"), "stdout:\n{stdout}");
}

/// A session with no Tool 3 run renders the empty message, not an error.
#[test]
fn view_empty_session_says_no_findings() {
    let out = cl()
        .arg("divergence-view")
        .arg(example("minimal.cls.json"))
        .output()
        .expect("spawn cl");

    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("No divergence findings"),
        "stdout:\n{stdout}"
    );
}
