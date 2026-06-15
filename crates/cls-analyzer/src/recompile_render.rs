//! Render Tool 1's findings into the three user-facing shapes (`markdown` / `json` / `text`).
//!
//! This is deliberately the *only* place output formatting lives — the analyzer
//! ([`crate::recompile::analyze`]) returns plain data ([`RecompileFindings`] /
//! [`RecompileDiff`]) with **no** embedded rendering (discipline **D2**: findings are data,
//! presentation is a separate concern). The CLI picks a [`Format`] from `--format` and calls
//! one of the two entry points here; nothing else formats recompile output.
//!
//! Two render targets, mirroring the two analysis modes:
//! - [`render_findings`] — the single-session summary (`cl recompile-summary session.cls.json`).
//! - [`render_diff`] — the session-to-session regression diff (ADR-033, `--baseline`).
//!
//! `json` is `serde_json::to_string_pretty` over the findings struct (the struct *is* the
//! machine-readable schema for the analysis result); `markdown` and `text` are hand-rendered
//! so the human-facing wording is owned here rather than derived from field names.

use serde::Serialize;

use crate::recompile::{
    GrownCluster, GuardCategory, RecompileDiff, RecompileFindings, Suggestion, ValueTransition,
};

/// Which shape to render. Mirrors the CLI's `--format` flag; kept clap-free so the analyzer
/// crate has no CLI dependency (the CLI maps its own enum onto this one).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Format {
    /// Human-readable Markdown — the default `cl recompile-summary` output.
    Markdown,
    /// Machine-readable pretty JSON — the findings struct serialized verbatim.
    Json,
    /// Plain layout for terminals that don't render Markdown: no `#` headings or box-drawing
    /// in the renderer's own furniture, and ASCII `->` value arrows. (Suggestion *content*
    /// from the analyzer is passed through verbatim rather than transliterated.)
    Text,
}

/// Render a single-session [`RecompileFindings`] in the requested format.
pub fn render_findings(findings: &RecompileFindings, format: Format) -> String {
    match format {
        Format::Markdown => findings_markdown(findings),
        Format::Json => to_json(findings),
        Format::Text => findings_text(findings),
    }
}

/// Render a session-to-session [`RecompileDiff`] (regression mode, ADR-033) in the
/// requested format.
pub fn render_diff(diff: &RecompileDiff, format: Format) -> String {
    match format {
        Format::Markdown => diff_markdown(diff),
        Format::Json => to_json(diff),
        Format::Text => diff_text(diff),
    }
}

// ── JSON ────────────────────────────────────────────────────────────────────────────────

/// Pretty-print any findings struct as JSON. Infallible in practice — these are plain
/// owned structs with no maps-with-non-string-keys and no custom `Serialize`, so
/// `to_string_pretty` cannot fail; the `unwrap_or_else` keeps the signature `-> String`
/// rather than leaking a `Result` the caller can't meaningfully act on.
fn to_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize findings: {e}\"}}"))
}

// ── shared formatting helpers ─────────────────────────────────────────────────────────────

/// The tree connector for item `i` of `n`: `└` for the last item, `├` otherwise.
fn connector(i: usize, n: usize) -> char {
    if i + 1 == n {
        '└'
    } else {
        '├'
    }
}

/// `x[0]` / `x` / `(uncategorized)` — the axis a cluster attributes to, for display.
fn axis_label(axis: &Option<String>) -> &str {
    axis.as_deref().unwrap_or("(uncategorized)")
}

/// ` — values 4→8, 8→16` (up to 5 shown, then `, …`); empty string when the guard carried
/// no values (bool/value guards don't, per the v0.5.0 schema's optional fields).
fn values_clause(values: &[ValueTransition], arrow: &str) -> String {
    if values.is_empty() {
        return String::new();
    }
    let shown: Vec<String> = values
        .iter()
        .take(5)
        .map(|v| {
            format!(
                "{}{arrow}{}",
                v.previous.as_deref().unwrap_or("?"),
                v.new.as_deref().unwrap_or("?")
            )
        })
        .collect();
    let more = if values.len() > 5 { ", …" } else { "" };
    format!(" — values {}{more}", shown.join(", "))
}

// ── findings: Markdown ──────────────────────────────────────────────────────────────────

fn findings_markdown(f: &RecompileFindings) -> String {
    let mut out = String::from("## Recompile Summary\n\n");

    if f.total_recompilations == 0 {
        out.push_str("No recompilations recorded — the compiled region was cache-stable.\n");
        return out;
    }

    out.push_str(&format!(
        "Total recompilations: {}\n",
        f.total_recompilations
    ));
    out.push_str(&format!(
        "Triggered by: {} distinct guard {}\n",
        f.guard_categories.len(),
        plural(f.guard_categories.len(), "category", "categories"),
    ));

    let n = f.guard_categories.len();
    for (i, cat) in f.guard_categories.iter().enumerate() {
        out.push_str(&format!(
            "  {} {}\n",
            connector(i, n),
            category_line_md(cat)
        ));
    }

    if !f.top_suggestions.is_empty() {
        out.push('\n');
        out.push_str("### Top suggestions\n\n");
        out.push_str(&suggestions_md(&f.top_suggestions));
    }

    out
}

/// `size · \`x[0]\` (32 events — values 8→16, 16→24)` — one cluster on one Markdown line.
fn category_line_md(cat: &GuardCategory) -> String {
    format!(
        "{} · `{}` ({} {}{})",
        cat.category,
        axis_label(&cat.axis),
        cat.count,
        plural(cat.count as usize, "event", "events"),
        values_clause(&cat.observed_values, "→"),
    )
}

fn suggestions_md(suggestions: &[Suggestion]) -> String {
    let mut out = String::new();
    for (i, s) in suggestions.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", i + 1, s.text));
        out.push_str(&format!("   _evidence: {}_\n", s.evidence));
    }
    out
}

// ── findings: text ──────────────────────────────────────────────────────────────────────

fn findings_text(f: &RecompileFindings) -> String {
    let mut out = String::from("Recompile Summary\n=================\n");

    if f.total_recompilations == 0 {
        out.push_str("No recompilations recorded.\n");
        return out;
    }

    out.push_str(&format!(
        "Total recompilations: {}\n",
        f.total_recompilations
    ));
    out.push_str(&format!("Guard categories: {}\n", f.guard_categories.len()));
    for cat in &f.guard_categories {
        out.push_str(&format!("  - {}\n", category_line_text(cat)));
    }

    if !f.top_suggestions.is_empty() {
        out.push_str("Top suggestions:\n");
        for (i, s) in f.top_suggestions.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", i + 1, s.text));
            out.push_str(&format!("     evidence: {}\n", s.evidence));
        }
    }

    out
}

/// `size  x[0]  (32 events; values 8->16, 16->24)` — ASCII-only cluster line for `text`.
fn category_line_text(cat: &GuardCategory) -> String {
    format!(
        "{}  {}  ({} {}{})",
        cat.category,
        axis_label(&cat.axis),
        cat.count,
        plural(cat.count as usize, "event", "events"),
        values_clause(&cat.observed_values, "->").replace(" — ", "; "),
    )
}

// ── diff: Markdown ──────────────────────────────────────────────────────────────────────

fn diff_markdown(d: &RecompileDiff) -> String {
    let mut out = String::from("## Recompile Diff (vs baseline)\n\n");

    if d.added.is_empty() && d.grown.is_empty() && d.removed.is_empty() {
        out.push_str("No recompile differences vs baseline — cache behaviour unchanged.\n");
        return out;
    }

    out.push_str(&format!(
        "Added: {} · Grown: {} · Removed: {}\n",
        d.added.len(),
        d.grown.len(),
        d.removed.len(),
    ));

    md_cluster_section(
        &mut out,
        "Added — new recompiles this change introduced",
        &d.added,
    );
    md_grown_section(&mut out, &d.grown);
    md_cluster_section(&mut out, "Removed — no longer recompiling", &d.removed);

    if !d.suggestions.is_empty() {
        out.push('\n');
        out.push_str("### Suggestions\n\n");
        out.push_str(&suggestions_md(&d.suggestions));
    }

    out
}

fn md_cluster_section(out: &mut String, heading: &str, clusters: &[GuardCategory]) {
    out.push('\n');
    out.push_str(&format!("### {heading}\n"));
    if clusters.is_empty() {
        out.push_str("  (none)\n");
        return;
    }
    let n = clusters.len();
    for (i, cat) in clusters.iter().enumerate() {
        out.push_str(&format!(
            "  {} {}\n",
            connector(i, n),
            category_line_md(cat)
        ));
    }
}

fn md_grown_section(out: &mut String, grown: &[GrownCluster]) {
    out.push('\n');
    out.push_str("### Grown — recompiled more than baseline\n");
    if grown.is_empty() {
        out.push_str("  (none)\n");
        return;
    }
    let n = grown.len();
    for (i, g) in grown.iter().enumerate() {
        out.push_str(&format!(
            "  {} {} — was {} in baseline\n",
            connector(i, n),
            category_line_md(&g.cluster),
            g.base_count,
        ));
    }
}

// ── diff: text ──────────────────────────────────────────────────────────────────────────

fn diff_text(d: &RecompileDiff) -> String {
    let mut out = String::from("Recompile Diff (vs baseline)\n============================\n");

    if d.added.is_empty() && d.grown.is_empty() && d.removed.is_empty() {
        out.push_str("No recompile differences vs baseline.\n");
        return out;
    }

    out.push_str(&format!(
        "Added: {}  Grown: {}  Removed: {}\n",
        d.added.len(),
        d.grown.len(),
        d.removed.len(),
    ));

    text_cluster_section(&mut out, "Added", &d.added);
    out.push_str("Grown:\n");
    if d.grown.is_empty() {
        out.push_str("  (none)\n");
    } else {
        for g in &d.grown {
            out.push_str(&format!(
                "  - {} (was {})\n",
                category_line_text(&g.cluster),
                g.base_count,
            ));
        }
    }
    text_cluster_section(&mut out, "Removed", &d.removed);

    if !d.suggestions.is_empty() {
        out.push_str("Suggestions:\n");
        for (i, s) in d.suggestions.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", i + 1, s.text));
            out.push_str(&format!("     evidence: {}\n", s.evidence));
        }
    }

    out
}

fn text_cluster_section(out: &mut String, heading: &str, clusters: &[GuardCategory]) {
    out.push_str(&format!("{heading}:\n"));
    if clusters.is_empty() {
        out.push_str("  (none)\n");
        return;
    }
    for cat in clusters {
        out.push_str(&format!("  - {}\n", category_line_text(cat)));
    }
}

// ── tiny shared util ──────────────────────────────────────────────────────────────────────

/// Pick the singular or plural noun for `n`.
fn plural<'a>(n: usize, one: &'a str, many: &'a str) -> &'a str {
    if n == 1 {
        one
    } else {
        many
    }
}
