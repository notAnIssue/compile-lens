//! Integration tests for the wired `cl fusion-detect` subcommand (Tool 6).
//!
//! These spawn the built `cl` binary. The populated cases use a temp `.cls.json` fixture carrying a
//! Pattern A (`GEMM-Residual-RMSNorm-GEMM`) graph with captured shapes — so the matcher, shape
//! derivation, cost model, and renderer run end-to-end across the process boundary. The empty case
//! uses the committed `minimal.cls.json` (no Pattern A). `fusion-detect` is suggest-only and
//! read-only, so every successful read exits 0 (it is informational, not a CI gate).

use std::path::PathBuf;
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

/// Write a `.cls.json` fixture to a uniquely-named temp file and return its path.
fn fixture(name: &str, json: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("cl_fusion_detect_{name}.cls.json"));
    std::fs::write(&path, json).expect("write temp fixture");
    path
}

const SESSION_HEAD: &str = r#""schema_version":"0.5.0",
"session":{"id":"00000000-0000-4000-8000-000000000000","timestamp":"t",
           "torch_version":"2.6.0","redaction_policy":"default-strict"}"#;

/// One native Pattern A on the canonical LLaMA shape (M=8192, N=11008, K0=4096, K1=4096, bf16) →
/// one opportunity at ~2.04×, high confidence.
fn one_pattern_a() -> String {
    format!(
        r#"{{{SESSION_HEAD},
        "compiled_graphs":[{{"graph_id":"g","nodes":[
            {{"id":"x","op_type":"placeholder","out_shape":[8192,4096],"out_dtype":"bf16"}},
            {{"id":"w0","op_type":"placeholder","out_shape":[4096,11008],"out_dtype":"bf16"}},
            {{"id":"bias","op_type":"placeholder","out_shape":[11008],"out_dtype":"bf16"}},
            {{"id":"resid","op_type":"placeholder","out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"gamma","op_type":"placeholder","out_shape":[11008],"out_dtype":"bf16"}},
            {{"id":"w1","op_type":"placeholder","out_shape":[11008,4096],"out_dtype":"bf16"}},
            {{"id":"g1","op_type":"aten.addmm","inputs":["bias","x","w0"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"add","op_type":"aten.add","inputs":["g1","resid"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"rms","op_type":"aten.rms_norm","inputs":["add","gamma"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"g2","op_type":"aten.mm","inputs":["rms","w1"],"out_shape":[8192,4096],"out_dtype":"bf16"}}
        ]}}]
    }}"#
    )
}

/// Two native Pattern A layers (shared weight/input placeholders) → two opportunities.
fn two_pattern_a() -> String {
    format!(
        r#"{{{SESSION_HEAD},
        "compiled_graphs":[{{"graph_id":"g","nodes":[
            {{"id":"x","op_type":"placeholder","out_shape":[8192,4096],"out_dtype":"bf16"}},
            {{"id":"w0","op_type":"placeholder","out_shape":[4096,11008],"out_dtype":"bf16"}},
            {{"id":"bias","op_type":"placeholder","out_shape":[11008],"out_dtype":"bf16"}},
            {{"id":"resid","op_type":"placeholder","out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"gamma","op_type":"placeholder","out_shape":[11008],"out_dtype":"bf16"}},
            {{"id":"w1","op_type":"placeholder","out_shape":[11008,4096],"out_dtype":"bf16"}},
            {{"id":"g1_0","op_type":"aten.addmm","inputs":["bias","x","w0"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"add_0","op_type":"aten.add","inputs":["g1_0","resid"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"rms_0","op_type":"aten.rms_norm","inputs":["add_0","gamma"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"g2_0","op_type":"aten.mm","inputs":["rms_0","w1"],"out_shape":[8192,4096],"out_dtype":"bf16"}},
            {{"id":"g1_1","op_type":"aten.addmm","inputs":["bias","x","w0"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"add_1","op_type":"aten.add","inputs":["g1_1","resid"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"rms_1","op_type":"aten.rms_norm","inputs":["add_1","gamma"],"out_shape":[8192,11008],"out_dtype":"bf16"}},
            {{"id":"g2_1","op_type":"aten.mm","inputs":["rms_1","w1"],"out_shape":[8192,4096],"out_dtype":"bf16"}}
        ]}}]
    }}"#
    )
}

#[test]
fn fusion_detect_markdown_reports_the_opportunity() {
    let path = fixture("md", &one_pattern_a());
    let out = cl()
        .arg("fusion-detect")
        .arg(&path)
        .output()
        .expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("## Fusion Opportunities Detected"),
        "stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("GEMM-Residual-RMSNorm-GEMM"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("~2.04×"), "stdout:\n{stdout}");
    assert!(stdout.contains("Confidence: high"), "stdout:\n{stdout}");
}

#[test]
fn fusion_detect_json_is_valid_and_schema_aligned() {
    let path = fixture("json", &one_pattern_a());
    let out = cl()
        .arg("fusion-detect")
        .arg(&path)
        .arg("--format")
        .arg("json")
        .output()
        .expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    let arr = value["fusion_opportunities"]
        .as_array()
        .expect("schema key");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["pattern_id"], "A");
    // JSON carries the full-precision ratio (markdown is what rounds to ~2.04×).
    let speedup = arr[0]["estimated_speedup"]
        .as_f64()
        .expect("speedup is a number");
    assert!(
        (speedup - 2.04).abs() < 0.01,
        "speedup ~2.04, got {speedup}"
    );
}

#[test]
fn fusion_detect_defaults_to_markdown() {
    let path = fixture("default", &one_pattern_a());
    let out = cl()
        .arg("fusion-detect")
        .arg(&path)
        .output()
        .expect("spawn cl");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("## Fusion Opportunities Detected"),
        "stdout:\n{stdout}"
    );
}

#[test]
fn fusion_detect_no_opportunities_is_clean() {
    // minimal.cls.json carries no Pattern A → the empty message, exit 0 (not an error).
    let out = cl()
        .arg("fusion-detect")
        .arg(example("minimal.cls.json"))
        .output()
        .expect("spawn cl");
    assert!(out.status.success(), "exit: {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("No fusion opportunities detected."),
        "stdout:\n{stdout}"
    );
}

#[test]
fn fusion_detect_min_speedup_filters_below_threshold() {
    // The lone opportunity is ~2.04×; a 100× floor drops it → empty message, still exit 0.
    let path = fixture("minspeedup", &one_pattern_a());
    let out = cl()
        .arg("fusion-detect")
        .arg(&path)
        .arg("--min-speedup")
        .arg("100")
        .output()
        .expect("spawn cl");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("No fusion opportunities detected."),
        "stdout:\n{stdout}"
    );
}

#[test]
fn fusion_detect_top_k_caps_the_report() {
    // Two opportunities in the graph; --top-k 1 keeps exactly one (one "### #" header).
    let path = fixture("topk", &two_pattern_a());
    let out = cl()
        .arg("fusion-detect")
        .arg(&path)
        .arg("--top-k")
        .arg("1")
        .output()
        .expect("spawn cl");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let headers = stdout.matches("### #").count();
    assert_eq!(
        headers, 1,
        "top-k 1 must keep one opportunity; stdout:\n{stdout}"
    );
}

#[test]
fn fusion_detect_help_states_suggest_only() {
    let out = cl()
        .arg("fusion-detect")
        .arg("--help")
        .output()
        .expect("spawn cl");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Suggest-only") || stdout.contains("suggest-only"),
        "help must state suggest-only; stdout:\n{stdout}"
    );
}
