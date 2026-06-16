//! Tests for the hero HTML report renderer.

use cls_schema::ClsArtifact;

fn from_json(json: &str) -> ClsArtifact {
    serde_json::from_str(json).expect("artifact parses")
}

/// A committed example artifact, addressed relative to this crate's manifest dir.
fn example(name: &str) -> ClsArtifact {
    let path = format!(
        "{}/../../schema/examples/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    from_json(&std::fs::read_to_string(path).expect("read example"))
}

const MINIMAL_SESSION: &str = r#"{"schema_version":"0.5.0","session":{"id":"s1",
"timestamp":"2026-01-01T00:00:00Z","torch_version":"2.6.0","redaction_policy":"default-strict",
"gpu_name":"A100-SXM-80GB"}}"#;

#[test]
fn renders_a_self_contained_offline_document() {
    let html = cls_report::render(&from_json(MINIMAL_SESSION));
    assert!(html.starts_with("<!DOCTYPE html>"), "missing doctype");
    assert!(html.contains("<style>"), "CSS must be inlined");
    // Offline: no external/CDN references anywhere in the document.
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "no external URLs allowed"
    );
    assert!(html.contains("Session metadata"));
    assert!(html.contains("2.6.0"), "torch version shown");
    assert!(html.contains("A100-SXM-80GB"), "gpu shown");
}

#[test]
fn a_cache_stable_session_says_so() {
    // minimal.cls.json has no recompilations.
    let html = cls_report::render(&example("minimal.cls.json"));
    assert!(html.contains("Recompile summary"));
    assert!(html.contains("cache-stable"));
}

#[test]
fn renders_the_recompilation_from_the_full_example() {
    // full.cls.json carries one recompilation.
    let html = cls_report::render(&example("full.cls.json"));
    assert!(html.contains("Recompile summary"));
    assert!(html.contains("recompilation(s)"));
    assert!(html.contains("<table>"));
    assert!(html.contains("Raw artifacts"));
}

#[test]
fn user_controlled_strings_are_escaped() {
    // A torch_version carrying an injection must render as inert text, never live markup.
    let html = cls_report::render(&from_json(
        r#"{"schema_version":"0.5.0","session":{"id":"s1","timestamp":"t",
        "torch_version":"<script>alert(1)</script>","redaction_policy":"default-strict"}}"#,
    ));
    assert!(
        !html.contains("<script>alert(1)</script>"),
        "injection must not survive verbatim"
    );
    assert!(html.contains("&lt;script&gt;"), "must be escaped");
}
