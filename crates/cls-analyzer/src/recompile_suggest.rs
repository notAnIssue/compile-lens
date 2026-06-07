//! Turn guard clusters into ranked, axis-precise suggestions (the "prescribe" step).
//!
//! [`crate::recompile_cluster`] attributed each recompile to a `(category, axis)`; this module
//! turns each *actionable* cluster into one [`Suggestion`] whose advice is precise to that axis
//! — `mark_dynamic(x, 0)`, not a vague "consider marking something dynamic". It is **N2**: a
//! rejectable hint with an evidence trace back to the cluster, never an auto-applied patch.
//!
//! One confident rule per category we can name with certainty (`size` / `dtype` / `stride`); the
//! `other` bucket (unrecognized guards) yields **no** suggestion rather than a vague one — a
//! noisy tool gets disabled. Clusters arrive already ordered by descending recompile count, so
//! the suggestions inherit that "most-impactful-first" ranking.

use crate::recompile::{GrownCluster, GuardCategory, Suggestion, ValueTransition};

/// Rank-ordered suggestions, one per actionable cluster (skipping `other`).
pub fn suggest(clusters: &[GuardCategory]) -> Vec<Suggestion> {
    clusters.iter().filter_map(suggest_for_cluster).collect()
}

/// Regression-anchored suggestions for a session diff (ADR-033): the same fix as the
/// single-snapshot case, but framed as "this change *introduced* / *worsened* it" — which the
/// before/after diff makes defensible. Added clusters first (the new regressions), then grown.
pub fn suggest_regression(added: &[GuardCategory], grown: &[GrownCluster]) -> Vec<Suggestion> {
    let mut out = Vec::new();
    for cluster in added {
        if let Some(fix) = suggest_for_cluster(cluster) {
            out.push(Suggestion {
                text: format!(
                    "This change introduced {} recompile(s) on a new axis — {}",
                    cluster.count, fix.text
                ),
                evidence: format!("ADDED vs baseline · {}", fix.evidence),
            });
        }
    }
    for g in grown {
        if let Some(fix) = suggest_for_cluster(&g.cluster) {
            out.push(Suggestion {
                text: format!(
                    "This change worsened recompiles ({} → {}) — {}",
                    g.base_count, g.cluster.count, fix.text
                ),
                evidence: format!(
                    "GROWN vs baseline ({} → {}) · {}",
                    g.base_count, g.cluster.count, fix.evidence
                ),
            });
        }
    }
    out
}

/// Split an axis label (`"x[0]"` / `"x"`) into `(tensor, optional_dim)`.
fn split_axis(axis: &str) -> (&str, Option<&str>) {
    match axis.split_once('[') {
        Some((tensor, rest)) => (tensor, rest.strip_suffix(']')),
        None => (axis, None),
    }
}

/// A compact ` (values 4→8, 8→16)` trace for the evidence string; empty when no values.
fn value_summary(values: &[ValueTransition]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let shown: Vec<String> = values
        .iter()
        .take(5)
        .map(|v| {
            format!(
                "{}→{}",
                v.previous.as_deref().unwrap_or("?"),
                v.new.as_deref().unwrap_or("?")
            )
        })
        .collect();
    let more = if values.len() > 5 { ", …" } else { "" };
    format!(" (values {}{more})", shown.join(", "))
}

pub(crate) fn suggest_for_cluster(cluster: &GuardCategory) -> Option<Suggestion> {
    // `other`-category clusters have no extractable axis -> no confident suggestion (N2).
    let axis = cluster.axis.as_deref()?;
    let (tensor, dim) = split_axis(axis);

    let text = match cluster.category.as_str() {
        "size" => {
            let dim = dim?; // a size guard always names a dim; bail defensively if not
            format!(
                "`{tensor}` dim {dim} keeps changing size — mark it dynamic to stop the \
                 recompiles: `torch._dynamo.mark_dynamic({tensor}, {dim})` (or compile with \
                 `dynamic=True`)."
            )
        }
        "dtype" => format!(
            "`{tensor}`'s dtype changes across calls, forcing a fresh compile each time — pin \
             it (cast once) upstream of the compiled region."
        ),
        "stride" => format!(
            "`{tensor}` is non-contiguous on some calls — insert `{tensor} = \
             {tensor}.contiguous()` before the compiled region so its layout is stable."
        ),
        _ => return None, // unrecognized guard family -> no suggestion
    };

    let evidence = format!(
        "{} cluster on {axis} — {} recompile(s){}",
        cluster.category,
        cluster.count,
        value_summary(&cluster.observed_values)
    );
    Some(Suggestion { text, evidence })
}
