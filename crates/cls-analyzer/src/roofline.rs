//! Tool 5 analysis: compose the three-layer roofline cost model over a session's kernels.
//!
//! `cls-roofline` holds the model itself — Layer 1 (the Williams theoretical lower bound), Layer 2
//! (the empirical predictor's four corrections), and Layer 3 (rank-correlation-gated pruning). This
//! module is the *artifact-level* analysis on top of it: for each kernel in a `.cls.json` that
//! carries the features Layer 1 needs, it produces a full [`RooflinePrediction`]
//! (`RooflineCostModel::predict` = Layer 1 + Layer 2). When at least two kernels also carry a
//! measured runtime, it calibrates the predicted ranking against the measured one (Spearman) and
//! computes a grid-level [`PruningDecision`] (Layer 3) — how many of the configs an autotuner would
//! actually have to measure.
//!
//! The result is a [`RooflineReport`], which the CLI renders as markdown or JSON. The autotune
//! harness drives this same path out-of-process (`cl kernel-roofline --format json`), which is why
//! the report is a plain serializable struct rather than something CLI-specific.

use serde::Serialize;

use cls_errors::ClsError;
use cls_roofline::{pruning_decision, RooflineCostModel};
use cls_schema::{Calibration, ClsArtifact, PruningDecision, RooflinePrediction};

/// The Tool 5 analysis result: per-kernel predictions, plus a grid-level calibration and pruning
/// decision when the session carries measured runtimes.
#[derive(Debug, Clone, Serialize)]
pub struct RooflineReport {
    /// The GPU spec the predictions were computed against.
    pub gpu: String,
    /// One prediction per kernel that had the features Layer 1 needs (flops + bytes).
    pub predictions: Vec<RooflinePrediction>,
    /// Predicted-vs-measured calibration across the kernels that carried a measured runtime.
    /// `None` when fewer than two such kernels exist — a correlation needs at least two points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
    /// The grid-level pruning decision (which prune tier, and how many configs to measure). `None`
    /// without enough paired predicted/measured points to calibrate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pruning_decision: Option<PruningDecision>,
}

/// Run Tool 5 over a session against a named GPU, targeting `top_k` configs under aggressive
/// pruning. `Err` only if the GPU name is not in the registry.
pub fn analyze(
    artifact: &ClsArtifact,
    gpu_name: &str,
    top_k: usize,
) -> Result<RooflineReport, ClsError> {
    let model = RooflineCostModel::for_gpu(gpu_name).ok_or_else(|| ClsError::InvalidCliArgs {
        detail: format!("unknown GPU `{gpu_name}`; not in the cls-roofline spec registry"),
    })?;

    let mut predictions = Vec::new();
    // Paired predicted/measured series, in kernel order, for the grid-level calibration.
    let mut predicted = Vec::new();
    let mut measured = Vec::new();

    for kernel in &artifact.kernels {
        let Some(features) = &kernel.features else {
            continue; // no features -> Layer 1 can't draw the roofline; skip this kernel
        };
        let Some(mut prediction) = model.predict(&kernel.kernel_id, features) else {
            continue; // features present but missing flops/bytes -> still no roofline
        };

        // Attach the measured runtime (the proton / self-timed mean) when present, and pair it with
        // the predictor so the grid can be calibrated.
        if let Some(mean_us) = kernel.measurements.as_ref().and_then(|m| m.mean_us) {
            prediction.measured_us = Some(mean_us);
            if let Some(predictor_us) = prediction.empirical_predictor_us {
                if mean_us > 0.0 {
                    // How close the predictor came: predictor / measured (the schema example's 0.94).
                    prediction.achieved_vs_predictor = Some(predictor_us / mean_us);
                }
                predicted.push(predictor_us);
                measured.push(mean_us);
            }
        }

        predictions.push(prediction);
    }

    // Layer 3 needs at least two paired points to correlate; `pruning_decision` is itself `None` if
    // the series can't be correlated (constant, or mismatched), in which case so is the calibration.
    let (calibration, pruning) = match predicted.len() {
        n if n >= 2 => {
            let decision = pruning_decision(&predicted, &measured, top_k);
            let calibration = decision.as_ref().map(|d| Calibration {
                pearson_correlation: None,
                spearman_correlation: d.threshold_rank_correlation,
                calibration_verdict: d.mode.as_deref().map(calibration_verdict),
                n_samples: Some(n as u64),
            });
            (calibration, decision)
        }
        _ => (None, None),
    };

    Ok(RooflineReport {
        gpu: gpu_name.to_string(),
        predictions,
        calibration,
        pruning_decision: pruning,
    })
}

/// A human verdict for the calibration, derived from the prune tier the rank correlation produced.
fn calibration_verdict(mode: &str) -> String {
    match mode {
        "aggressive" => "reliable",
        "moderate" => "partial",
        _ => "unreliable",
    }
    .to_string()
}

/// Output format for the report.
#[derive(Debug, Clone, Copy)]
pub enum Format {
    Markdown,
    Json,
}

/// Render a report as pretty JSON (machine-readable, what the harness consumes) or markdown.
pub fn render(report: &RooflineReport, format: Format) -> String {
    match format {
        Format::Json => serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string()),
        Format::Markdown => render_markdown(report),
    }
}

fn render_markdown(report: &RooflineReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "## Kernel Roofline ({})\n", report.gpu);

    if report.predictions.is_empty() {
        let _ = writeln!(out, "_No kernels with roofline features in this session._");
        return out;
    }

    let _ = writeln!(
        out,
        "| kernel | bound | lower µs | predicted µs | measured µs | corrections |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|");
    for p in &report.predictions {
        let corrections = if p.corrections_applied.is_empty() {
            "—".to_string()
        } else {
            p.corrections_applied.join(", ")
        };
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} |",
            p.kernel_id,
            p.bound_type.as_deref().unwrap_or("—"),
            fmt_us(p.theoretical_lower_bound_us),
            fmt_us(p.empirical_predictor_us),
            fmt_us(p.measured_us),
            corrections,
        );
    }

    if let Some(pd) = &report.pruning_decision {
        let _ = writeln!(
            out,
            "\n**Pruning:** mode `{}` — measure {} of {} configs (rank correlation {}).",
            pd.mode.as_deref().unwrap_or("—"),
            opt_u64(pd.configs_measured),
            opt_u64(pd.configs_total),
            pd.threshold_rank_correlation
                .map(|r| format!("{r:.3}"))
                .unwrap_or_else(|| "—".to_string()),
        );
    }
    out
}

fn fmt_us(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}"))
        .unwrap_or_else(|| "—".to_string())
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "—".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an artifact from a `kernels[]` JSON literal (deserializing keeps the test robust to
    /// ClsArtifact's many other fields).
    fn artifact(kernels_json: &str) -> ClsArtifact {
        let json = format!(
            r#"{{"schema_version":"0.5.0",
                 "session":{{"id":"00000000-0000-4000-8000-000000000000",
                             "timestamp":"2026-01-01T00:00:00Z",
                             "torch_version":"2.6.0",
                             "redaction_policy":"default-strict"}},
                 "kernels":{kernels_json}}}"#
        );
        serde_json::from_str(&json).expect("artifact parses")
    }

    // Three configs of one workload (same flops/bytes), block_size shrinking 256 -> 64 -> 32 so the
    // predictor rises monotonically (block_size penalty); measured runtimes rise in the same order,
    // so the predicted-vs-measured rank correlation is perfect.
    const THREE_CONFIGS: &str = r#"[
        {"kernel_id":"cfg_256","name":"k","features":{"flops":1e12,"bytes_loaded":1e10,"bytes_stored":0,"block_size":256,"num_regs":32,"n_spills":0},"measurements":{"mean_us":100.0,"source":"proton"}},
        {"kernel_id":"cfg_64","name":"k","features":{"flops":1e12,"bytes_loaded":1e10,"bytes_stored":0,"block_size":64,"num_regs":32,"n_spills":0},"measurements":{"mean_us":130.0,"source":"proton"}},
        {"kernel_id":"cfg_32","name":"k","features":{"flops":1e12,"bytes_loaded":1e10,"bytes_stored":0,"block_size":32,"num_regs":32,"n_spills":0},"measurements":{"mean_us":145.0,"source":"proton"}}
    ]"#;

    #[test]
    fn analyze_fills_predictions_and_attaches_measurements() {
        let report = analyze(&artifact(THREE_CONFIGS), "A100-SXM-80GB", 1).unwrap();
        assert_eq!(report.predictions.len(), 3);
        for p in &report.predictions {
            assert!(p.empirical_predictor_us.is_some(), "Layer 2 filled");
            assert!(p.theoretical_lower_bound_us.is_some(), "Layer 1 filled");
            assert!(p.measured_us.is_some(), "measured attached");
            assert!(
                p.achieved_vs_predictor.is_some(),
                "predictor/measured ratio"
            );
        }
        // predictor rises as block_size shrinks (the 256 config is the cheapest).
        let p256 = report.predictions[0].empirical_predictor_us.unwrap();
        let p32 = report.predictions[2].empirical_predictor_us.unwrap();
        assert!(
            p32 > p256,
            "smaller block predicted slower: {p32} vs {p256}"
        );
    }

    #[test]
    fn perfect_rank_correlation_yields_aggressive_pruning() {
        // top_k = 1: aggressive measures only the single best config of three (a 3x prune).
        let report = analyze(&artifact(THREE_CONFIGS), "A100-SXM-80GB", 1).unwrap();
        let pruning = report
            .pruning_decision
            .expect("paired points -> a decision");
        assert_eq!(pruning.mode.as_deref(), Some("aggressive"));
        assert_eq!(pruning.configs_total, Some(3));
        assert_eq!(pruning.configs_measured, Some(1));
        let cal = report.calibration.expect("calibration present");
        assert!((cal.spearman_correlation.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(cal.calibration_verdict.as_deref(), Some("reliable"));
        assert_eq!(cal.n_samples, Some(3));
    }

    #[test]
    fn without_measurements_there_is_no_calibration_or_pruning() {
        let no_meas = r#"[
            {"kernel_id":"k0","name":"k","features":{"flops":1e12,"bytes_loaded":1e10,"bytes_stored":0,"block_size":256,"num_regs":32,"n_spills":0}}
        ]"#;
        let report = analyze(&artifact(no_meas), "A100-SXM-80GB", 5).unwrap();
        assert_eq!(report.predictions.len(), 1);
        assert!(report.predictions[0].empirical_predictor_us.is_some());
        assert!(report.predictions[0].measured_us.is_none());
        assert!(report.calibration.is_none());
        assert!(report.pruning_decision.is_none());
    }

    #[test]
    fn kernels_without_roofline_features_are_skipped() {
        let mixed = r#"[
            {"kernel_id":"has_features","name":"k","features":{"flops":1e12,"bytes_loaded":1e10,"bytes_stored":0}},
            {"kernel_id":"no_features","name":"k"}
        ]"#;
        let report = analyze(&artifact(mixed), "A100-SXM-80GB", 5).unwrap();
        assert_eq!(report.predictions.len(), 1);
        assert_eq!(report.predictions[0].kernel_id, "has_features");
    }

    #[test]
    fn unknown_gpu_is_an_error() {
        let err = analyze(&artifact(THREE_CONFIGS), "no-such-gpu", 5).unwrap_err();
        assert!(matches!(err, ClsError::InvalidCliArgs { .. }));
    }

    #[test]
    fn json_render_round_trips_and_markdown_has_a_table() {
        let report = analyze(&artifact(THREE_CONFIGS), "A100-SXM-80GB", 1).unwrap();
        let json = render(&report, Format::Json);
        let back: serde_json::Value = serde_json::from_str(&json).expect("json parses back");
        assert_eq!(back["gpu"], "A100-SXM-80GB");
        assert_eq!(back["predictions"].as_array().unwrap().len(), 3);
        assert_eq!(back["pruning_decision"]["mode"], "aggressive");

        let md = render(&report, Format::Markdown);
        assert!(md.contains("## Kernel Roofline (A100-SXM-80GB)"));
        assert!(md.contains("| kernel | bound |"));
        assert!(md.contains("**Pruning:**"));
    }
}
