//! Match-coverage benchmark (design discipline D7, todo invariant).
//!
//! Runs `diff_graphs` on real captured PyTorch small-model graph pairs (a model and a minor
//! variant) and asserts the median `match_coverage` clears 0.70. This guards that the matcher
//! actually covers most nodes on realistic graphs — not just the hand-built synthetic corpus. If
//! the median falls below 0.70 that is a release blocker, not a threshold to lower.

use std::fs;
use std::path::Path;

use cls_schema::ClsArtifact;
use cls_wl_diff::{diff_graphs, FxGraph};

/// The captured real-model pairs, under `tests/fixtures/diff/real/<name>/`. Each is a small model
/// diffed against a minor variant (depth, activation, width, residual, bias).
const PAIRS: &[&str] = &["depth", "activation", "width", "residual", "bias"];

const THRESHOLD: f64 = 0.70;

fn graph(pair: &str, side: &str) -> FxGraph {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/diff/real")
        .join(pair)
        .join(side);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let artifact: ClsArtifact =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    FxGraph::from_nodes(&artifact.compiled_graphs[0].nodes)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

#[test]
fn test_median_match_coverage_clears_threshold() {
    let coverages: Vec<f64> = PAIRS
        .iter()
        .map(|pair| {
            let d = diff_graphs(&graph(pair, "base.cls.json"), &graph(pair, "head.cls.json"));
            println!("{pair}: match_coverage = {:.3}", d.match_coverage);
            d.match_coverage
        })
        .collect();

    let median = median(coverages.clone());
    println!(
        "median match_coverage = {median:.3} over {} pairs",
        coverages.len()
    );
    assert!(coverages.len() >= 5, "need at least 5 model pairs");
    assert!(
        median >= THRESHOLD,
        "median match_coverage {median:.3} is below the {THRESHOLD} release-blocker threshold; \
         investigate the matcher or the fixtures, do not lower the threshold"
    );
}
