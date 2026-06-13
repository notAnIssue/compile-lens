//! View-only rendering of Tool 3's `divergences[]` (ADR-034).
//!
//! This is the read side of `cl divergence-view`: it takes the divergence findings already
//! captured in a `.cls.json` and renders them, with **no torch and no recompilation** — the eager
//! vs compiled run happened when the artifact was written, so viewing a prior session is a pure
//! read-and-format of the stored records.

use std::fmt::Write as _;

use cls_schema::Divergence;

/// Output format. `Text` is folded into `Markdown` by the CLI, so only two shapes exist here.
#[derive(Clone, Copy, Debug)]
pub enum Format {
    Markdown,
    Json,
}

/// Render a session's divergence findings in the chosen format.
pub fn render(divergences: &[Divergence], format: Format) -> String {
    match format {
        // A slice of derive-`Serialize` records of owned data cannot fail to serialize.
        Format::Json => {
            serde_json::to_string_pretty(divergences).expect("Divergence records serialize")
        }
        Format::Markdown => render_markdown(divergences),
    }
}

fn render_markdown(divergences: &[Divergence]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Divergence");

    if divergences.is_empty() {
        let _ = writeln!(out, "\nNo divergence findings in this session.");
        return out;
    }

    for d in divergences {
        let _ = writeln!(out);
        match &d.first_divergent_layer {
            // A finding that compared layers and found none past tolerance — the clean case.
            None => {
                let _ = writeln!(
                    out,
                    "- `{}`: no divergence — {} layers compared within tolerance (rtol={}, atol={}).",
                    d.divergence_id, d.num_layers_compared, d.rtol, d.atol
                );
            }
            Some(layer) => {
                let _ = writeln!(
                    out,
                    "### {} — first divergent layer `{layer}`",
                    d.divergence_id
                );
                match d.max_abs_diff {
                    Some(diff) => {
                        let _ = writeln!(out, "- Max abs diff: {diff}");
                    }
                    None => {
                        let _ = writeln!(out, "- Shape mismatch (max abs diff undefined)");
                    }
                }
                let _ = writeln!(
                    out,
                    "- Compared {} layers (rtol={}, atol={})",
                    d.num_layers_compared, d.rtol, d.atol
                );
                if let Some(cause) = &d.suggested_cause {
                    let _ = writeln!(out, "- Suggested cause: {cause}");
                }
                if let Some(attr) = &d.attribution {
                    let passes = if attr.responsible_passes.is_empty() {
                        "none".to_string()
                    } else {
                        attr.responsible_passes.join(", ")
                    };
                    let _ = writeln!(
                        out,
                        "- Attribution (pass-level): attributed={}, passes: {passes}, {} probes",
                        attr.attributed, attr.num_probes
                    );
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cls_schema::DivergenceAttribution;

    fn attributed_finding() -> Divergence {
        Divergence {
            divergence_id: "dv_001".into(),
            first_divergent_layer: Some("decoder.layers.3".into()),
            max_abs_diff: Some(0.297),
            num_layers_compared: 48,
            rtol: 1e-3,
            atol: 1e-5,
            suggested_cause: Some("disabled: post_grad_custom_post_pass".into()),
            attribution: Some(DivergenceAttribution {
                attributed: true,
                responsible_passes: vec!["post_grad_custom_post_pass".into()],
                summary: "disabled: post_grad_custom_post_pass".into(),
                num_probes: 8,
            }),
        }
    }

    fn clean_finding() -> Divergence {
        Divergence {
            divergence_id: "dv_clean".into(),
            first_divergent_layer: None,
            max_abs_diff: None,
            num_layers_compared: 12,
            rtol: 1e-3,
            atol: 1e-5,
            suggested_cause: None,
            attribution: None,
        }
    }

    #[test]
    fn markdown_lists_a_divergence_with_layer_and_attribution() {
        let md = render(&[attributed_finding()], Format::Markdown);
        assert!(md.contains("## Divergence"), "missing heading:\n{md}");
        assert!(md.contains("decoder.layers.3"), "missing layer:\n{md}");
        assert!(
            md.contains("post_grad_custom_post_pass"),
            "missing pass:\n{md}"
        );
        assert!(md.contains("0.297"), "missing diff:\n{md}");
    }

    #[test]
    fn markdown_empty_says_so() {
        let md = render(&[], Format::Markdown);
        assert!(md.contains("No divergence findings"), "empty:\n{md}");
    }

    #[test]
    fn markdown_clean_finding_says_no_divergence() {
        let md = render(&[clean_finding()], Format::Markdown);
        assert!(md.contains("no divergence"), "clean:\n{md}");
        assert!(md.contains("12 layers"), "clean count:\n{md}");
    }

    #[test]
    fn json_round_trips() {
        let json = render(&[attributed_finding()], Format::Json);
        let parsed: Vec<Divergence> =
            serde_json::from_str(&json).expect("rendered json parses back");
        assert_eq!(parsed.len(), 1);
        assert_eq!(
            parsed[0].first_divergent_layer.as_deref(),
            Some("decoder.layers.3")
        );
        assert_eq!(
            parsed[0].attribution.as_ref().unwrap().responsible_passes,
            vec!["post_grad_custom_post_pass"]
        );
    }
}
