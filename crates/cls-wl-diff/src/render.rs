//! Render an [`IrGraphDiff`] to the three output formats `cl diff --format` exposes.
//!
//! The diff itself is computed by [`crate::diff_graphs`]; this is the presentation layer, kept
//! separate so the algorithm stays pure. `json` serializes the whole struct (the machine-readable
//! contract); `markdown` and `text` are human summaries with the added / removed / modified counts,
//! the per-node detail, and the two quality numbers.

use crate::diff::IrGraphDiff;

/// Output format for a rendered diff, mirroring `cl diff --format markdown|json|text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Markdown,
    Json,
    Text,
}

/// Render `diff` to `format`. Returns the full document as a string (the CLI prints it verbatim).
pub fn render(diff: &IrGraphDiff, format: Format) -> String {
    match format {
        Format::Json => render_json(diff),
        Format::Markdown => render_markdown(diff),
        Format::Text => render_text(diff),
    }
}

/// The machine-readable contract: the `IrGraphDiff` serialized verbatim (pretty-printed).
fn render_json(diff: &IrGraphDiff) -> String {
    // serde_json on a derive-Serialize struct of owned strings/floats cannot fail.
    serde_json::to_string_pretty(diff).expect("IrGraphDiff serializes")
}

fn render_markdown(diff: &IrGraphDiff) -> String {
    let mut out = String::new();
    out.push_str("## Compile Diff\n\n");
    out.push_str(&format!(
        "{} added · {} removed · {} modified · {} matched\n\n",
        diff.added.len(),
        diff.removed.len(),
        diff.modified.len(),
        diff.matched.len(),
    ));

    if !diff.added.is_empty() {
        out.push_str("### Added\n");
        for id in &diff.added {
            out.push_str(&format!("- `{id}`\n"));
        }
        out.push('\n');
    }
    if !diff.removed.is_empty() {
        out.push_str("### Removed\n");
        for id in &diff.removed {
            out.push_str(&format!("- `{id}`\n"));
        }
        out.push('\n');
    }
    if !diff.modified.is_empty() {
        out.push_str("### Modified\n");
        for (base, head) in &diff.modified {
            // base and head ids are usually equal; show both when they differ (e.g. a rename).
            if base == head {
                out.push_str(&format!("- `{base}`\n"));
            } else {
                out.push_str(&format!("- `{base}` → `{head}`\n"));
            }
        }
        out.push('\n');
    }
    if diff.added.is_empty() && diff.removed.is_empty() && diff.modified.is_empty() {
        out.push_str("No structural changes.\n\n");
    }

    out.push_str(&format!(
        "_match coverage {:.2} · anchor uniqueness {:.2}_\n",
        diff.match_coverage, diff.anchor_uniqueness_ratio,
    ));
    out
}

fn render_text(diff: &IrGraphDiff) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "compile diff: {} added, {} removed, {} modified, {} matched\n",
        diff.added.len(),
        diff.removed.len(),
        diff.modified.len(),
        diff.matched.len(),
    ));
    for id in &diff.added {
        out.push_str(&format!("  + {id}\n"));
    }
    for id in &diff.removed {
        out.push_str(&format!("  - {id}\n"));
    }
    for (base, head) in &diff.modified {
        if base == head {
            out.push_str(&format!("  ~ {base}\n"));
        } else {
            out.push_str(&format!("  ~ {base} -> {head}\n"));
        }
    }
    out.push_str(&format!(
        "match coverage {:.2}, anchor uniqueness {:.2}\n",
        diff.match_coverage, diff.anchor_uniqueness_ratio,
    ));
    out
}

#[cfg(test)]
mod tests {
    use cls_schema::FxNode;

    use super::*;
    use crate::{diff_graphs, FxGraph};

    fn node(id: &str, op: &str, inputs: &[&str]) -> FxNode {
        FxNode {
            id: id.into(),
            op_type: op.into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            attrs: Default::default(),
        }
    }

    /// A small add-a-node diff to render.
    fn sample() -> IrGraphDiff {
        let base = [
            node("p", "placeholder", &[]),
            node("a", "aten.relu.default", &["p"]),
        ];
        let head = [
            node("p", "placeholder", &[]),
            node("a", "aten.relu.default", &["p"]),
            node("b", "aten.abs.default", &["a"]),
        ];
        diff_graphs(&FxGraph::from_nodes(&base), &FxGraph::from_nodes(&head))
    }

    #[test]
    fn test_markdown_lists_the_added_node_and_quality() {
        let md = render(&sample(), Format::Markdown);
        assert!(md.contains("## Compile Diff"));
        assert!(md.contains("### Added"));
        assert!(md.contains("`b`"));
        assert!(md.contains("match coverage"));
    }

    #[test]
    fn test_json_round_trips_to_the_same_struct() {
        let diff = sample();
        let json = render(&diff, Format::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["added"][0], "b");
        assert!(parsed["match_coverage"].is_number());
    }

    #[test]
    fn test_text_is_plain_and_lists_changes() {
        let txt = render(&sample(), Format::Text);
        assert!(txt.contains("1 added"));
        assert!(txt.contains("+ b"));
    }

    #[test]
    fn test_no_change_diff_says_so() {
        let g = [
            node("p", "placeholder", &[]),
            node("a", "aten.relu.default", &["p"]),
        ];
        let diff = diff_graphs(&FxGraph::from_nodes(&g), &FxGraph::from_nodes(&g));
        assert!(render(&diff, Format::Markdown).contains("No structural changes"));
    }
}
