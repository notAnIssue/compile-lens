//! `cls-report` — the hero HTML report (Phase 7).
//!
//! Renders a `.cls.json` session into one **self-contained, offline-browseable** HTML file: no CDN,
//! all CSS inline. It reuses the analyzers (P8 — wrap, don't re-implement): the recompile section
//! is rendered from [`cls_analyzer::recompile::analyze`]'s findings, not re-derived. Rendering emits
//! **fresh HTML** from the structured findings (not the markdown renderers), so every
//! user-controlled string is escaped with [`esc`] at the point it enters the document.
//!
//! Sections so far: session metadata, recompile summary, divergence (eager vs compiled, surfacing a
//! NaN `max_abs_diff` as the headline signal per ADR-039), fusion opportunities (CODA crown-jewel),
//! and a raw-artifacts footer. The remaining tool sections (compile-diff / cache-stability / lint /
//! roofline) and the security hardening (an `ammonia` allowlist + an XSS payload corpus + a CSP
//! header + a URL allowlist) land in later changes.

use cls_schema::{ClsArtifact, Session};

/// Render a session artifact into a self-contained HTML report.
pub fn render(artifact: &ClsArtifact) -> String {
    let mut sections = String::new();
    sections.push_str(&metadata_section(&artifact.session));
    sections.push_str(&recompile_section(artifact));
    sections.push_str(&divergence_section(artifact));
    sections.push_str(&fusion_section(artifact));
    sections.push_str(&raw_section(artifact));
    document(&artifact.session, &sections)
}

/// Escape the five HTML-significant characters in a user-controlled string. The full XSS pass (an
/// `ammonia` allowlist + a payload corpus + a CSP header) lands in a later change; escaping here
/// keeps every interpolated string correct-by-default in the meantime.
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

fn document(session: &Session, sections: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>compile-lens session report</title>\n<style>{CSS}</style>\n</head>\n\
         <body>\n<header>\n<h1>compile-lens session report</h1>\n\
         <p class=\"sub\">torch {} · {}</p>\n</header>\n<main>\n{sections}</main>\n</body>\n</html>\n",
        esc(&session.torch_version),
        esc(&session.timestamp),
    )
}

/// `<details open><summary>title</summary> body </details>` — a collapsible section.
fn section(id: &str, title: &str, body: &str) -> String {
    format!(
        "<details open id=\"{}\">\n<summary><h2>{}</h2></summary>\n\
         <div class=\"body\">{body}</div>\n</details>\n",
        esc(id),
        esc(title),
    )
}

/// One `<dt>/<dd>` row, only when the value is present.
fn row(label: &str, value: Option<&str>) -> String {
    match value {
        Some(v) => format!("<dt>{}</dt><dd>{}</dd>", esc(label), esc(v)),
        None => String::new(),
    }
}

fn metadata_section(s: &Session) -> String {
    let mut dl = String::from("<dl class=\"meta\">");
    dl.push_str(&row("torch", Some(&s.torch_version)));
    dl.push_str(&row("triton", s.triton_version.as_deref()));
    dl.push_str(&row("CUDA", s.cuda_version.as_deref()));
    dl.push_str(&row("python", s.python_version.as_deref()));
    dl.push_str(&row("GPU", s.gpu_name.as_deref()));
    dl.push_str(&row("host", s.host.as_deref()));
    dl.push_str(&row("command", s.command.as_deref()));
    dl.push_str(&row("session id", Some(&s.id)));
    dl.push_str("</dl>");
    section("metadata", "Session metadata", &dl)
}

fn recompile_section(artifact: &ClsArtifact) -> String {
    let findings = match cls_analyzer::recompile::analyze(artifact) {
        Ok(f) => f,
        // analyze only fails on malformed records; say so rather than render nothing.
        Err(_) => {
            return section(
                "recompile",
                "Recompile summary",
                "<p class=\"muted\">could not analyze recompilations</p>",
            );
        }
    };

    if findings.total_recompilations == 0 {
        return section(
            "recompile",
            "Recompile summary",
            "<p class=\"good\">No recompilations recorded — the compiled region was cache-stable.</p>",
        );
    }

    let mut body = format!(
        "<p><strong>{}</strong> recompilation(s), across <strong>{}</strong> guard categor(y/ies).</p>",
        findings.total_recompilations,
        findings.guard_categories.len(),
    );
    body.push_str(
        "<table>\n<thead><tr><th>category</th><th>count</th><th>axis</th>\
         <th>observed values</th></tr></thead>\n<tbody>\n",
    );
    for cat in &findings.guard_categories {
        let values: Vec<String> = cat
            .observed_values
            .iter()
            .take(5)
            .map(|v| {
                format!(
                    "{}→{}",
                    esc(v.previous.as_deref().unwrap_or("?")),
                    esc(v.new.as_deref().unwrap_or("?")),
                )
            })
            .collect();
        let more = if cat.observed_values.len() > 5 {
            ", …"
        } else {
            ""
        };
        body.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}{}</td></tr>\n",
            esc(&cat.category),
            cat.count,
            esc(cat.axis.as_deref().unwrap_or("(uncategorized)")),
            values.join(", "),
            more,
        ));
    }
    body.push_str("</tbody>\n</table>\n");

    if !findings.top_suggestions.is_empty() {
        body.push_str("<h3>Top suggestions</h3>\n<ol>\n");
        for s in &findings.top_suggestions {
            body.push_str(&format!(
                "<li>{}<br><span class=\"muted\">evidence: {}</span></li>\n",
                esc(&s.text),
                esc(&s.evidence),
            ));
        }
        body.push_str("</ol>\n");
    }
    section("recompile", "Recompile summary", &body)
}

fn divergence_section(artifact: &ClsArtifact) -> String {
    if artifact.divergences.is_empty() {
        return section(
            "divergence",
            "Divergence (eager vs compiled)",
            "<p class=\"muted\">No divergence findings recorded.</p>",
        );
    }
    let mut body = String::from(
        "<table>\n<thead><tr><th>first divergent layer</th><th>max abs diff</th>\
         <th>layers compared</th><th>attributed cause</th></tr></thead>\n<tbody>\n",
    );
    for d in &artifact.divergences {
        // max_abs_diff = NaN is the headline signal: the compiled model produced NaN (ADR-039).
        let max_diff = match d.max_abs_diff {
            Some(v) if v.is_nan() => "<strong>NaN</strong> — compiled output is NaN".to_string(),
            Some(v) if v.is_infinite() => "∞".to_string(),
            Some(v) => format!("{v:.3e}"),
            None => "—".to_string(),
        };
        body.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            esc(d
                .first_divergent_layer
                .as_deref()
                .unwrap_or("(none — within tolerance)")),
            max_diff,
            d.num_layers_compared,
            esc(d.suggested_cause.as_deref().unwrap_or("—")),
        ));
    }
    body.push_str("</tbody>\n</table>\n");
    section("divergence", "Divergence (eager vs compiled)", &body)
}

/// Render an optional `u64` dimension, `?` when absent.
fn dim(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string())
}

fn fusion_section(artifact: &ClsArtifact) -> String {
    if artifact.fusion_opportunities.is_empty() {
        return section(
            "fusion",
            "Fusion opportunities (CODA)",
            "<p class=\"muted\">No algebraic fusion opportunities found.</p>",
        );
    }
    let mut body = String::from(
        "<p class=\"muted\">Analytical HBM-traffic roofline (memory-bound upper bound, not \
         measured); suggest-only.</p>\n\
         <table>\n<thead><tr><th>pattern</th><th>shape (M×N×K0×K1)</th>\
         <th>HBM bytes: baseline → fused</th><th>est. speedup</th><th>suggested kernel</th>\
         </tr></thead>\n<tbody>\n",
    );
    for f in &artifact.fusion_opportunities {
        let shape = f.shape.as_ref().map_or_else(
            || "—".to_string(),
            |s| format!("{}×{}×{}×{}", dim(s.m), dim(s.n), dim(s.k0), dim(s.k1),),
        );
        let hbm = format!(
            "{} → {}",
            f.baseline_hbm_bytes
                .map_or_else(|| "?".to_string(), |b| format!("{b:.3e}")),
            f.fused_hbm_bytes
                .map_or_else(|| "?".to_string(), |b| format!("{b:.3e}")),
        );
        let speedup = f
            .estimated_speedup
            .map_or_else(|| "—".to_string(), |s| format!("{s:.2}×"));
        body.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td><strong>{}</strong></td>\
             <td><code>{}</code></td></tr>\n",
            esc(&f.pattern_id),
            esc(&shape),
            esc(&hbm),
            esc(&speedup),
            esc(f.suggested_kernel.as_deref().unwrap_or("—")),
        ));
    }
    body.push_str("</tbody>\n</table>\n");
    section("fusion", "Fusion opportunities (CODA)", &body)
}

fn raw_section(artifact: &ClsArtifact) -> String {
    let counts = format!(
        "<dl class=\"meta\">\
         <dt>schema</dt><dd>{}</dd>\
         <dt>recompilations</dt><dd>{}</dd>\
         <dt>compiled graphs</dt><dd>{}</dd>\
         <dt>graph breaks</dt><dd>{}</dd>\
         <dt>iterations</dt><dd>{}</dd>\
         </dl>\
         <p class=\"muted\">The machine-readable <code>.cls.json</code> is the source of truth for \
         this report.</p>",
        esc(&artifact.schema_version),
        artifact.recompilations.len(),
        artifact.compiled_graphs.len(),
        artifact.graph_breaks.len(),
        artifact.iterations.len(),
    );
    section("raw", "Raw artifacts", &counts)
}

const CSS: &str = "\
:root{color-scheme:light dark}\
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5;\
margin:0;padding:0 1rem 4rem;color:#1a1a1a;background:#fafafa}\
header{max-width:60rem;margin:0 auto;padding:1.5rem 0 .5rem}\
h1{font-size:1.4rem;margin:0}\
.sub{color:#666;margin:.25rem 0 0;font-size:.9rem}\
main{max-width:60rem;margin:0 auto}\
details{background:#fff;border:1px solid #e2e2e2;border-radius:8px;margin:1rem 0;padding:.5rem 1rem}\
summary{cursor:pointer;list-style:none}\
summary h2{display:inline;font-size:1.1rem;margin:0}\
.body{padding:.5rem 0}\
dl.meta{display:grid;grid-template-columns:max-content 1fr;gap:.2rem 1rem;margin:0}\
dl.meta dt{color:#666;font-size:.9rem}\
dl.meta dd{margin:0;font-variant-numeric:tabular-nums}\
table{border-collapse:collapse;width:100%;font-size:.9rem}\
th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #eee}\
th{color:#666;font-weight:600}\
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em}\
.muted{color:#888;font-size:.9rem}\
.good{color:#1a7f37}\
";
