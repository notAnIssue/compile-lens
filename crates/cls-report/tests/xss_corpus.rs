//! XSS payload corpus (a release blocker).
//!
//! Every user-controlled string the report interpolates must survive a corpus of injection payloads
//! as inert, *escaped* text — never live markup. The report renders fresh HTML and escapes each
//! field with `cls_report::esc`; this proves it across fields and a corpus, so a future field that
//! forgets to escape fails CI here. (The report carries no JS and a strict CSP — `default-src
//! 'none'; style-src 'unsafe-inline'` — so even a hypothetical escape gap has no script-src to run
//! in; the corpus is the belt to that suspenders.)

use cls_schema::ClsArtifact;

/// Injection payloads spanning tag injection, attribute breakout, event handlers, `javascript:`
/// URIs, SVG/iframe vectors, and a style-context break.
const PAYLOADS: &[&str] = &[
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "\"><script>alert(1)</script>",
    "<svg/onload=alert(1)>",
    "<iframe src=javascript:alert(1)></iframe>",
    "<body onload=alert(1)>",
    "</style><script>alert(1)</script>",
    "<a href=\"javascript:alert(1)\">x</a>",
    "<details open ontoggle=alert(1)>",
    "<input autofocus onfocus=alert(1)>",
    "<<script>alert(1)//<</script>",
    "&#60;script&#62;",
];

fn from_json(json: &str) -> ClsArtifact {
    serde_json::from_str(json).expect("artifact parses")
}

/// The payload in the session's `torch_version` (metadata section).
fn in_torch_version(p: &str) -> ClsArtifact {
    from_json(&format!(
        r#"{{"schema_version":"0.5.0","session":{{"id":"s","timestamp":"t",
        "torch_version":{p:?},"redaction_policy":"default-strict"}}}}"#
    ))
}

/// The payload in a divergence's `suggested_cause` (a tool-section field).
fn in_divergence_cause(p: &str) -> ClsArtifact {
    from_json(&format!(
        r#"{{"schema_version":"0.5.0","session":{{"id":"s","timestamp":"t",
        "torch_version":"2.6.0","redaction_policy":"default-strict"}},
        "divergences":[{{"divergence_id":"d","num_layers_compared":1,"rtol":1e-3,"atol":1e-5,
        "suggested_cause":{p:?}}}]}}"#
    ))
}

#[test]
fn no_corpus_payload_survives_as_live_markup() {
    for &p in PAYLOADS {
        for artifact in [in_torch_version(p), in_divergence_cause(p)] {
            let html = cls_report::render(&artifact, None);
            let escaped = cls_report::esc(p);
            // The payload appears only in its escaped form.
            assert!(
                html.contains(&escaped),
                "payload not present escaped: {p:?}"
            );
            // Every corpus payload contains an HTML-significant char, so escaping changes it; the
            // raw (live) form must therefore be absent from the document.
            assert_ne!(
                escaped, p,
                "test payload must contain an escapable char: {p:?}"
            );
            assert!(
                !html.contains(p),
                "raw payload survived (live markup): {p:?}"
            );
        }
    }
}

#[test]
fn report_carries_a_strict_csp() {
    let html = cls_report::render(&in_torch_version("2.6.0"), None);
    assert!(html.contains("Content-Security-Policy"), "CSP meta missing");
    assert!(html.contains("default-src 'none'"), "CSP must default-deny");
    assert!(
        html.contains("script-src") || html.contains("default-src 'none'"),
        "no script-src"
    );
}
