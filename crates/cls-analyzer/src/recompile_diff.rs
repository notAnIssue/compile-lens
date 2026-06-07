//! Session-to-session recompile diff (ADR-033) — the regression mode.
//!
//! Single-snapshot [`crate::recompile::analyze`] answers "this run recompiled N times".
//! [`diff_recompiles`] answers the **comparison-axis** question the ecosystem's snapshot tools
//! can't: "what did this commit/PR *change* about the recompile behaviour, vs a baseline?" —
//! e.g. *"this PR introduced 40 recompiles on `x[0]`"*. The diff is the ground truth, so its
//! regression-anchored suggestions are more defensible than free-floating advice.
//!
//! It reuses [`crate::recompile_cluster::cluster`] on each side, matches clusters by their
//! `(category, axis, canonical)` identity, and classifies into added / grown / removed.
//!
//! **Baseline-pairing seam (ADR-033 §5)**: this takes two already-parsed [`ClsArtifact`]s. The
//! *loading* of base + head from disk / a CLI `--baseline` flag is deliberately left to the
//! caller (CLI layer) so it can be shared with Tool 2a (compile-diff), which also needs to load
//! an artifact pair — neither tool should grow its own divergent baseline-loading path.

use std::collections::HashMap;

use cls_schema::ClsArtifact;

use crate::recompile::{GrownCluster, GuardCategory, RecompileDiff};
use crate::{recompile_cluster, recompile_suggest};

/// Identity of a cluster for matching across the two sessions: `(category, axis, canonical)`.
/// Within one session a `(category, axis)` maps to one canonical template, so this is stable;
/// `axis` is `None` for `other`-category guards, where the canonical carries the identity.
type ClusterKey = (String, Option<String>, String);

fn key(cluster: &GuardCategory) -> ClusterKey {
    (
        cluster.category.clone(),
        cluster.axis.clone(),
        cluster.canonical_guard.clone(),
    )
}

/// Diff two sessions' recompiles into added / grown / removed clusters + regression suggestions.
pub fn diff_recompiles(base: &ClsArtifact, head: &ClsArtifact) -> RecompileDiff {
    let base_clusters = recompile_cluster::cluster(&base.recompilations);
    let head_clusters = recompile_cluster::cluster(&head.recompilations);

    let base_by_key: HashMap<ClusterKey, &GuardCategory> =
        base_clusters.iter().map(|c| (key(c), c)).collect();

    // head_clusters arrive ordered by descending count, so added/grown inherit that ranking.
    let mut added = Vec::new();
    let mut grown = Vec::new();
    for head_cluster in &head_clusters {
        match base_by_key.get(&key(head_cluster)) {
            None => added.push(head_cluster.clone()),
            Some(base_cluster) if head_cluster.count > base_cluster.count => {
                grown.push(GrownCluster {
                    cluster: head_cluster.clone(),
                    base_count: base_cluster.count,
                });
            }
            Some(_) => {} // unchanged or shrank — not a regression
        }
    }

    let head_keys: std::collections::HashSet<ClusterKey> = head_clusters.iter().map(key).collect();
    let removed = base_clusters
        .iter()
        .filter(|c| !head_keys.contains(&key(c)))
        .cloned()
        .collect();

    let suggestions = recompile_suggest::suggest_regression(&added, &grown);
    RecompileDiff {
        added,
        grown,
        removed,
        suggestions,
    }
}
