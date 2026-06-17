//! Share-time sanitization for an HTML **report** (`cl scrub report.html`).
//!
//! The hero report is born safe: `cls-report` emits fresh HTML, escapes every interpolated value,
//! and stamps a strict `Content-Security-Policy` (`default-src 'none'`) so a browser runs no script
//! and loads no external resource. This module is the defense-in-depth pass for an HTML file you
//! might *not* have generated fresh (an older report, hand-edited HTML), to make it share-safe:
//!
//! 1. **Ensure the strict CSP** is present — this is the actual guarantee. Under `default-src
//!    'none'` no inline or external script executes, so even a `<script>` that slipped in is inert.
//! 2. **Re-scrub secret-looking tokens** from the visible text (an `hf_…` / `sk-…` / `AKIA…` that
//!    leaked into a rendered field), reusing the same argv token rules as the JSON scrub.
//! 3. **Neutralize literal dangerous constructs** (`<script>`, `on*=` handlers, `javascript:`
//!    URIs). This is cosmetic cleanup *behind* the CSP — the CSP is what makes it safe; removing the
//!    constructs just means the shared file does not even contain them.

use std::sync::LazyLock;

use regex::Regex;

use crate::rules;

/// The strict policy `cls-report` stamps and this scrub guarantees: no script, no external resource,
/// only inline styles. Kept byte-identical to the renderer's so a re-scrub is a no-op on a fresh
/// report.
const STRICT_CSP: &str =
    "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">";

/// `<script` / `</script` (case-insensitive) — neutralized by escaping the opening `<` so the tag
/// renders as visible text rather than an element.
static SCRIPT_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<(/?script)").expect("static script pattern is valid"));

/// An inline event-handler attribute (`onclick=…`, `onerror=…`). Matches the attribute and its
/// value (quoted or bare) so the whole thing can be dropped.
static EVENT_HANDLER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)"#)
        .expect("static handler pattern is valid")
});

/// A `javascript:` URI scheme (case-insensitive), in any attribute.
static JS_URI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)javascript:").expect("static js-uri pattern is valid"));

/// What an HTML scrub changed, for the operator-facing summary.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HtmlScrubStats {
    pub csp_added: bool,
    pub tokens_scrubbed: bool,
    pub scripts_neutralized: usize,
    pub handlers_neutralized: usize,
    pub js_uris_neutralized: usize,
}

impl HtmlScrubStats {
    /// Whether the document was already share-safe (nothing to do).
    pub fn is_noop(&self) -> bool {
        !self.csp_added
            && !self.tokens_scrubbed
            && self.scripts_neutralized == 0
            && self.handlers_neutralized == 0
            && self.js_uris_neutralized == 0
    }
}

/// Sanitize an HTML report for sharing. Returns the cleaned HTML and a summary of what changed.
/// Idempotent: a fresh `cls-report` document (strict CSP, escaped content, no script) comes back
/// byte-identical with an all-zero [`HtmlScrubStats`].
pub fn scrub_html(html: &str) -> (String, HtmlScrubStats) {
    let mut stats = HtmlScrubStats::default();

    // 1. Re-scrub secret tokens from the visible text (same rules as the argv scrub).
    let (mut out, tokens_scrubbed) = rules::scrub_command(html, false);
    stats.tokens_scrubbed = tokens_scrubbed;

    // 2. Neutralize dangerous constructs (cosmetic; the CSP is the guarantee).
    stats.scripts_neutralized = SCRIPT_TAG.find_iter(&out).count();
    if stats.scripts_neutralized > 0 {
        out = SCRIPT_TAG.replace_all(&out, "&lt;$1").into_owned();
    }
    stats.handlers_neutralized = EVENT_HANDLER.find_iter(&out).count();
    if stats.handlers_neutralized > 0 {
        out = EVENT_HANDLER.replace_all(&out, "").into_owned();
    }
    stats.js_uris_neutralized = JS_URI.find_iter(&out).count();
    if stats.js_uris_neutralized > 0 {
        out = JS_URI.replace_all(&out, "blocked:").into_owned();
    }

    // 3. Ensure the strict CSP is present — the actual XSS guarantee.
    if !out.contains("Content-Security-Policy") {
        out = inject_csp(&out);
        stats.csp_added = true;
    }

    (out, stats)
}

/// Insert the strict CSP meta tag immediately after `<head>` (case-insensitive), or — if there is
/// no head — at the very top so a browser still sees it before any content.
fn inject_csp(html: &str) -> String {
    static HEAD_OPEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)<head[^>]*>").expect("static head pattern is valid"));
    if let Some(m) = HEAD_OPEN.find(html) {
        let mut out = String::with_capacity(html.len() + STRICT_CSP.len() + 1);
        out.push_str(&html[..m.end()]);
        out.push('\n');
        out.push_str(STRICT_CSP);
        out.push_str(&html[m.end()..]);
        out
    } else {
        format!("{STRICT_CSP}\n{html}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_report_is_a_noop() {
        // A document already carrying the strict CSP and no dangerous constructs is untouched.
        let fresh = format!(
            "<!DOCTYPE html><html><head>{STRICT_CSP}</head><body><p>safe</p></body></html>"
        );
        let (out, stats) = scrub_html(&fresh);
        assert!(stats.is_noop(), "stats: {stats:?}");
        assert_eq!(
            out, fresh,
            "a born-safe report must round-trip byte-identical"
        );
    }

    #[test]
    fn injects_csp_when_missing() {
        let (out, stats) = scrub_html("<html><head><title>x</title></head><body></body></html>");
        assert!(stats.csp_added);
        assert!(out.contains("default-src 'none'"));
        // Injected right after <head>, before the title.
        assert!(out.find("Content-Security-Policy").unwrap() < out.find("<title>").unwrap());
    }

    #[test]
    fn neutralizes_script_and_handlers_and_js_uris() {
        let evil = "<html><head></head><body><script>steal()</script>\
                    <img src=x onerror=\"steal()\"><a href=\"javascript:steal()\">x</a></body></html>";
        let (out, stats) = scrub_html(evil);
        assert_eq!(stats.scripts_neutralized, 2, "<script> and </script>");
        assert_eq!(stats.handlers_neutralized, 1);
        assert_eq!(stats.js_uris_neutralized, 1);
        assert!(
            !out.contains("<script"),
            "no live script tag remains: {out}"
        );
        assert!(!out.to_lowercase().contains("onerror="));
        assert!(!out.to_lowercase().contains("javascript:"));
        assert!(
            out.contains("&lt;script"),
            "script tag rendered as inert text"
        );
    }

    #[test]
    fn scrubs_leaked_tokens_in_text() {
        let leaky = "<html><head></head><body><code>--hf-token=hf_abcdefghijklmnopqrst</code></body></html>";
        let (out, stats) = scrub_html(leaky);
        assert!(stats.tokens_scrubbed);
        assert!(out.contains("--hf-token=<scrubbed>"));
        assert!(!out.contains("hf_abcdefghijklmnopqrst"));
    }

    #[test]
    fn scrub_html_is_idempotent() {
        let evil =
            "<html><head></head><body><script>x()</script><b onclick=\"y()\">z</b></body></html>";
        let (once, _) = scrub_html(evil);
        let (twice, stats2) = scrub_html(&once);
        assert!(stats2.is_noop(), "second pass changes nothing: {stats2:?}");
        assert_eq!(once, twice);
    }
}
