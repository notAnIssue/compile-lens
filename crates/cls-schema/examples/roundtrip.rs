//! Cross-language round-trip harness for the schema contract test.
//!
//! Reads a `.cls.json`, deserializes it into [`cls_schema::ClsArtifact`], re-serializes,
//! and writes the result. Used by `tests/integration/test_schema_round_trip.py` to prove
//! the Rust (serde) and Python (pydantic) bindings agree on schema v0.5.0.
//!
//! Usage: `cargo run --package cls-schema --example roundtrip -- <input.json> <output.json>`
//!
//! This is the same file + subprocess control-flow the toolkit uses (the design doc ADR-006):
//! no FFI — data crosses the language boundary only as a JSON file.

use std::process::ExitCode;

fn run(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(input)?;
    let artifact: cls_schema::ClsArtifact = serde_json::from_str(&text)?;
    let serialized = serde_json::to_string_pretty(&artifact)?;
    std::fs::write(output, serialized)?;
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: roundtrip <input.cls.json> <output.cls.json>");
        return ExitCode::from(2);
    };
    match run(input, output) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("roundtrip failed: {err}");
            ExitCode::FAILURE
        }
    }
}
