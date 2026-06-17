//! `cls-report` — the hero HTML report (Phase 7).
//!
//! Renders a `.cls.json` session into one **self-contained, offline-browseable** HTML file: no CDN,
//! all CSS inline. It reuses the analyzers (P8 — wrap, don't re-implement): the recompile section
//! is rendered from [`cls_analyzer::recompile::analyze`]'s findings, not re-derived. Rendering emits
//! **fresh HTML** from the structured findings (not the markdown renderers), so every
//! user-controlled string is escaped with [`esc`] at the point it enters the document.
//!
//! Sections: session metadata, the base→head IR diff (Tool 2a, present only when a baseline is
//! given), recompile summary, cache stability (Tool 2b), divergence (eager vs compiled, surfacing a
//! NaN `max_abs_diff` as the headline signal per ADR-039), lint (Tool 4), fusion opportunities
//! (CODA crown-jewel), roofline (Tool 5), and a raw-artifacts footer. The externally-computed
//! analyses (diff, DB-escalated lint, roofline) arrive via [`ReportInputs`]; everything else is
//! derived from the artifact. The XSS hardening (a payload corpus + a CSP header) is in place.

use cls_analyzer::lint::LintReport;
use cls_analyzer::roofline::RooflineReport;
use cls_schema::{ClsArtifact, Session};
// Re-exported: `ReportInputs` names it, so callers (the CLI, tests) need to as well.
pub use cls_wl_diff::IrGraphDiff;

/// The optional, externally-computed analyses a report can carry. Each is produced by the CLI — the
/// compile diff, the DB-escalated lint, the roofline against a GPU — and handed in, so the renderer
/// stays a pure function of what it is given (the same separation the diff already follows).
/// `ReportInputs::default()` is a bare single-session report.
#[derive(Default)]
pub struct ReportInputs<'a> {
    /// The base→head compile diff (Tool 2a); present in `--base` mode, renders the pillar section.
    pub diff: Option<&'a IrGraphDiff>,
    /// DB-escalated lint findings (Tool 4). `None` falls back to the artifact's raw candidates.
    pub lint: Option<&'a LintReport>,
    /// The roofline analysis (Tool 5), computed against the chosen GPU.
    pub roofline: Option<&'a RooflineReport>,
}

/// Render a session artifact into a self-contained HTML report. The optional analyses in `inputs`
/// (diff / lint / roofline) render as sections when present; pass `&ReportInputs::default()` for a
/// bare single-session report.
pub fn render(artifact: &ClsArtifact, inputs: &ReportInputs) -> String {
    let mut sections = String::new();
    sections.push_str(&metadata_section(&artifact.session));
    if let Some(d) = inputs.diff {
        sections.push_str(&ir_diff_section(d));
    }
    sections.push_str(&recompile_section(artifact));
    sections.push_str(&cache_stability_section(artifact));
    sections.push_str(&divergence_section(artifact));
    sections.push_str(&lint_section(artifact, inputs.lint));
    sections.push_str(&fusion_section(artifact));
    sections.push_str(&roofline_section(inputs.roofline));
    sections.push_str(&raw_section(artifact));
    document(&artifact.session, &sections)
}

/// Escape the five HTML-significant characters in a user-controlled string. This is the report's
/// XSS front line: the renderer emits fresh HTML and never keeps user HTML, so every interpolated
/// string passes through here. A payload corpus and a strict CSP back it up (see the XSS tests).
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
         <meta http-equiv=\"Content-Security-Policy\" \
         content=\"default-src 'none'; style-src 'unsafe-inline'\">\n\
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

/// The base→head compile diff (Tool 2a) — the report's pillar in `--base` mode. Renders the added /
/// removed / modified nodes plus two quality gauges (match coverage, anchor uniqueness) so the
/// reader can weigh how much to trust the structural verdict.
fn ir_diff_section(diff: &IrGraphDiff) -> String {
    let (added, removed, modified) = (diff.added.len(), diff.removed.len(), diff.modified.len());
    let headline = if added == 0 && removed == 0 && modified == 0 {
        "<p class=\"good\">No structural change — the compiled graph is identical between base and \
         head.</p>"
            .to_string()
    } else {
        format!(
            "<p class=\"warn\">The compiled graph changed: <strong>{added}</strong> added, \
             <strong>{removed}</strong> removed, <strong>{modified}</strong> modified node(s).</p>"
        )
    };
    let mut quality = String::from("<dl class=\"meta\">");
    quality.push_str(&row("match coverage", Some(&pct(diff.match_coverage))));
    quality.push_str(&row(
        "anchor uniqueness",
        Some(&pct(diff.anchor_uniqueness_ratio)),
    ));
    quality.push_str("</dl>");
    let lists = format!(
        "{}{}{}",
        node_list("added", &diff.added),
        node_list("removed", &diff.removed),
        modified_table(diff),
    );
    section(
        "ir-diff",
        "IR Diff (base → head)",
        &format!("{headline}{quality}{lists}"),
    )
}

/// A `<ul>` of node ids under an `<h3>` count heading, or nothing when the slice is empty.
fn node_list(label: &str, nodes: &[String]) -> String {
    if nodes.is_empty() {
        return String::new();
    }
    let items: String = nodes
        .iter()
        .map(|n| format!("<li><code>{}</code></li>", esc(n)))
        .collect();
    format!(
        "<h3>{} ({})</h3>\n<ul>{items}</ul>\n",
        esc(label),
        nodes.len()
    )
}

/// Modified `(base, head)` pairs as a table, annotated with the matcher's `[0, 1]` confidence.
fn modified_table(diff: &IrGraphDiff) -> String {
    if diff.modified.is_empty() {
        return String::new();
    }
    let confidence: std::collections::HashMap<&str, f64> = diff
        .matched
        .iter()
        .map(|(base, _head, c)| (base.as_str(), *c))
        .collect();
    let mut rows = String::new();
    for (base, head) in &diff.modified {
        let c = confidence.get(base.as_str()).copied().unwrap_or(0.0);
        rows.push_str(&format!(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{c:.2}</td></tr>\n",
            esc(base),
            esc(head),
        ));
    }
    format!(
        "<h3>modified ({})</h3>\n<table>\n<thead><tr><th>base</th><th>head</th>\
         <th>confidence</th></tr></thead>\n<tbody>\n{rows}</tbody>\n</table>\n",
        diff.modified.len(),
    )
}

/// Format a `[0, 1]` ratio as a whole-number percentage.
fn pct(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

/// Tool 4 lint. With a correctness DB the CLI escalates candidates to severity-rated findings and
/// passes them in (`Some`); without one, this falls back to the artifact's **raw candidates** and
/// says how to escalate — honest degradation, never a fabricated severity.
fn lint_section(artifact: &ClsArtifact, lint: Option<&LintReport>) -> String {
    match lint {
        Some(report) if report.findings.is_empty() => section(
            "lint",
            "Lint (Tool 4)",
            "<p class=\"good\">No correctness-lint findings survived escalation against the \
             database.</p>",
        ),
        Some(report) => {
            let mut body = String::from(
                "<p class=\"muted\">Static-scan candidates escalated against the correctness \
                 database — each carries a cited issue (lint-not-oracle).</p>\n\
                 <table>\n<thead><tr><th>severity</th><th>pattern</th><th>location</th>\
                 <th>issue</th><th>workaround</th></tr></thead>\n<tbody>\n",
            );
            for f in &report.findings {
                body.push_str(&format!(
                    "<tr><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                    severity_badge(&f.severity),
                    esc(&f.pattern_category),
                    location(f.source_location.as_ref()),
                    issue_ref(f.reference_issue.as_ref()),
                    esc(f.workaround.as_deref().unwrap_or("—")),
                ));
            }
            body.push_str("</tbody>\n</table>\n");
            section("lint", "Lint (Tool 4)", &body)
        }
        None if artifact.lint_findings.is_empty() => section(
            "lint",
            "Lint (Tool 4)",
            "<p class=\"good\">No lint candidates from the static scan.</p>",
        ),
        None => {
            let mut body = format!(
                "<p class=\"warn\">{} candidate pattern(s) from the static scan — not yet \
                 escalated. Pass <code>--db &lt;correctness.toml&gt;</code> to rate severity and \
                 attach issue links.</p>\n\
                 <table>\n<thead><tr><th>pattern</th><th>location</th><th>trigger</th>\
                 <th>confidence</th></tr></thead>\n<tbody>\n",
                artifact.lint_findings.len(),
            );
            for c in &artifact.lint_findings {
                body.push_str(&format!(
                    "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>\n",
                    esc(&c.pattern_category),
                    location(c.source_location.as_ref()),
                    esc(c.trigger_pattern.as_deref().unwrap_or("—")),
                    esc(c.confidence.as_deref().unwrap_or("—")),
                ));
            }
            body.push_str("</tbody>\n</table>\n");
            section("lint", "Lint (Tool 4)", &body)
        }
    }
}

/// Tool 5 roofline. Always computed against a GPU (the CLI defaults to the project baseline), so the
/// report shows real numbers out of the box; `None` only when the analysis was not run.
fn roofline_section(roofline: Option<&RooflineReport>) -> String {
    let Some(report) = roofline else {
        return section(
            "roofline",
            "Roofline (Tool 5)",
            "<p class=\"muted\">Roofline not computed.</p>",
        );
    };
    if report.predictions.is_empty() {
        return section(
            "roofline",
            "Roofline (Tool 5)",
            &format!(
                "<p class=\"muted\">No kernel carried the flops/bytes features the roofline model \
                 needs (GPU: <code>{}</code>).</p>",
                esc(&report.gpu),
            ),
        );
    }
    let mut body = format!(
        "<p class=\"muted\">Williams roofline + empirical predictor per kernel, against \
         <code>{}</code>.</p>\n\
         <table>\n<thead><tr><th>kernel</th><th>intensity (FLOP/B)</th><th>bound</th>\
         <th>predicted µs</th><th>measured µs</th></tr></thead>\n<tbody>\n",
        esc(&report.gpu),
    );
    for p in &report.predictions {
        body.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            esc(&p.kernel_id),
            num1(p.arithmetic_intensity),
            esc(p.bound_type.as_deref().unwrap_or("—")),
            num1(p.empirical_predictor_us),
            num1(p.measured_us),
        ));
    }
    body.push_str("</tbody>\n</table>\n");
    if let Some(verdict) = report
        .calibration
        .as_ref()
        .and_then(|c| c.calibration_verdict.as_deref())
    {
        body.push_str(&format!(
            "<p class=\"muted\">calibration: {}</p>\n",
            esc(verdict)
        ));
    }
    section("roofline", "Roofline (Tool 5)", &body)
}

/// `high` reads as a warning, `info` as muted; anything else passes through escaped.
fn severity_badge(severity: &str) -> String {
    match severity {
        "high" => "<strong class=\"warn\">high</strong>".to_string(),
        "info" => "<span class=\"muted\">info</span>".to_string(),
        other => esc(other),
    }
}

/// A `file:line` location as inline code, or `—` when absent.
fn location(loc: Option<&cls_schema::SourceRange>) -> String {
    match loc {
        Some(r) => {
            let file = esc(r.file.as_deref().unwrap_or("?"));
            match r.line_start {
                Some(line) => format!("<code>{file}:{line}</code>"),
                None => format!("<code>{file}</code>"),
            }
        }
        None => "—".to_string(),
    }
}

/// The reference issue URL as inert escaped text (a live link awaits the URL allowlist), or `—`.
fn issue_ref(issue: Option<&cls_schema::ReferenceIssue>) -> String {
    match issue.and_then(|i| i.url.as_deref()) {
        Some(url) => format!("<code>{}</code>", esc(url)),
        None => "—".to_string(),
    }
}

/// A microsecond figure to one decimal, or `—` when not measured/predicted.
fn num1(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.1}"),
        None => "—".to_string(),
    }
}

fn cache_stability_section(artifact: &ClsArtifact) -> String {
    let findings = cls_analyzer::cache_stability::analyze(artifact);
    if findings.findings.is_empty() {
        return section(
            "cache-stability",
            "Cache stability",
            "<p class=\"good\">No silently-stale-cache findings — recompilation tracked the state \
             changes.</p>",
        );
    }
    let mut body = String::from(
        "<p class=\"muted\">Iterations where module state drifted but the compiled graph was reused \
         and the output stayed frozen — the cache key missed the state change (Tool 2b).</p>\n\
         <table>\n<thead><tr><th>severity</th><th>iteration</th><th>drifted attrs</th>\
         <th>reference</th></tr></thead>\n<tbody>\n",
    );
    for f in &findings.findings {
        // Severity has a single variant (High) by construction; render it directly.
        body.push_str(&format!(
            "<tr><td><strong>high</strong></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>\n",
            f.iteration_index,
            esc(&f.changed_attrs.join(", ")),
            esc(&f.reference),
        ));
    }
    body.push_str("</tbody>\n</table>\n");
    section("cache-stability", "Cache stability", &body)
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
.warn{color:#9a6700}\
h3{font-size:.95rem;margin:.8rem 0 .3rem;color:#444}\
";
