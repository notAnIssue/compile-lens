//! `cls-scrub` — share-time redaction for a `.cls.json` artifact.
//!
//! The collector already writes a `default-strict` artifact clean (discipline D11, the Python
//! `security/redactor.py`). `cls-scrub` is the *post-hoc* companion: it takes an artifact at
//! any policy level and **promotes** it to a stricter one before sharing — re-applying the
//! redaction rules for the target level and, crucially, **refusing to demote** (redaction is a
//! one-way, lossy transform; raw data can only come back by re-collecting).
//!
//! Two layers:
//! - [`rules`] — the pure primitives (path / argv / host / env), a 1:1 port of the Python
//!   collect-time redactor so both halves agree byte-for-byte.
//! - this module — the policy layer: the strictness ordering, the promote-only guard, and
//!   [`redact_artifact`] which walks a typed [`ClsArtifact`] and rewrites every sensitive field.
//!
//! Spec: `docs/06_security/redaction_policy.md`.

pub mod html;
pub mod rules;

use cls_errors::ClsError;
use cls_schema::{ClsArtifact, RedactionPolicy};

/// Strictness rank: higher = more redacted. The promote-only rule is just "rank never
/// decreases". Order matches `redaction_policy.md` §4
/// (`confidential` → `internal` → `default-strict` → `public-safe`).
fn strictness(policy: RedactionPolicy) -> u8 {
    match policy {
        RedactionPolicy::Confidential => 0,
        RedactionPolicy::Internal => 1,
        RedactionPolicy::DefaultStrict => 2,
        RedactionPolicy::PublicSafe => 3,
    }
}

/// Whether the policy redacts sensitive fields by default. `default-strict` and `public-safe`
/// are share-safe and auto-scrub; `internal` / `confidential` deliberately keep raw fields
/// (opt-in / trusted), so promoting *to* them only relabels — there is nothing to scrub.
pub fn is_strict(policy: RedactionPolicy) -> bool {
    matches!(
        policy,
        RedactionPolicy::DefaultStrict | RedactionPolicy::PublicSafe
    )
}

/// The stricter (more redacted) of two policies. Used to pick a default scrub target that is
/// "at least `default-strict`" without ever demoting an artifact that is already stricter.
pub fn stricter_of(a: RedactionPolicy, b: RedactionPolicy) -> RedactionPolicy {
    if strictness(a) >= strictness(b) {
        a
    } else {
        b
    }
}

/// The kebab-case wire label for a policy. Public so callers (the `cl scrub` CLI) print the same
/// level names the user types as `--to`, and `CLS-E0009` messages read like the CLI flag.
pub fn level_name(policy: RedactionPolicy) -> &'static str {
    label(policy)
}

/// The kebab-case wire label for a policy (used in `CLS-E0009` messages so they read like the
/// CLI flag the user would type).
fn label(policy: RedactionPolicy) -> &'static str {
    match policy {
        RedactionPolicy::DefaultStrict => "default-strict",
        RedactionPolicy::Internal => "internal",
        RedactionPolicy::Confidential => "confidential",
        RedactionPolicy::PublicSafe => "public-safe",
    }
}

/// What a scrub actually changed, for the operator-facing summary line
/// (`✅ Wrote … (12 paths normalized, 1 argv token scrubbed, 47 kernel sources removed)`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RedactionStats {
    pub paths_normalized: usize,
    pub argv_tokens_scrubbed: bool,
    pub host_redacted: bool,
    pub kernel_sources_removed: usize,
    pub source_excerpts_removed: usize,
    pub env_vars_dropped: usize,
}

/// Promote `artifact` to the `target` redaction policy in place, returning a [`RedactionStats`]
/// summary of what changed.
///
/// `salt` is the per-install secret for the FQDN host hash (so the same machine hashes to a
/// stable `dh-<8hex>`); `strict_tokens` turns on the extra long-string argv scrub (§2.2,
/// `cl scrub --strict`).
///
/// Errors:
/// - [`ClsError::RedactionPolicyDemotionRefused`] (`CLS-E0009`) if `target` is *less* strict
///   than the artifact's current policy — demotion is refused outright.
/// - [`ClsError::SensitivePathDetected`] (`CLS-E0008`) if a path under a credential directory
///   is encountered; the scrub refuses rather than emit it even redacted.
pub fn redact_artifact(
    artifact: &mut ClsArtifact,
    target: RedactionPolicy,
    salt: &str,
    strict_tokens: bool,
) -> Result<RedactionStats, ClsError> {
    let source = artifact.session.redaction_policy;
    if strictness(target) < strictness(source) {
        return Err(ClsError::RedactionPolicyDemotionRefused {
            from: label(source).to_string(),
            to: label(target).to_string(),
        });
    }

    let mut stats = RedactionStats::default();

    // `internal` / `confidential` keep raw fields by design, so promoting to them only
    // relabels. Only the two share-safe levels actually rewrite fields.
    if is_strict(target) {
        let public_safe = matches!(target, RedactionPolicy::PublicSafe);
        let detected_repo = rules::detect_repo();
        let repo = detected_repo.as_deref();

        // ── session.command (§2.2 / level table) ──
        // public-safe nulls the command outright (its level-table treatment); default-strict
        // token-scrubs it but keeps the visible structure (`python {repo}/train.py …`).
        if public_safe {
            if artifact.session.command.take().is_some() {
                stats.argv_tokens_scrubbed = true;
            }
        } else if let Some(command) = artifact.session.command.as_ref() {
            let (scrubbed, changed) = rules::scrub_command(command, strict_tokens);
            artifact.session.command = Some(scrubbed);
            stats.argv_tokens_scrubbed = changed;
        }

        // ── session.host (§2.5) ──
        // public-safe nulls it; default-strict replaces the FQDN with a stable per-machine hash.
        // A host that is *already* a `dh-<hash>` (a re-scrub) is left untouched so repeated
        // scrubs converge — hashing the hash would drift on every pass.
        if let Some(host) = artifact.session.host.as_ref() {
            if public_safe {
                artifact.session.host = None;
                stats.host_redacted = true;
            } else if !rules::is_hashed_host(host) {
                artifact.session.host = Some(rules::hash_host(host, salt));
                stats.host_redacted = true;
            }
        }

        // ── env vars (§2.6) ──
        if let Some(env) = artifact.session.env_snapshot.as_mut() {
            let before = env.relevant_env_vars.len();
            env.relevant_env_vars
                .retain(|name, _| rules::env_var_allowed(name));
            stats.env_vars_dropped += before - env.relevant_env_vars.len();
        }

        // ── compiled-graph artifact paths (§2.1) ──
        for graph in artifact.compiled_graphs.iter_mut() {
            normalize_field(&mut graph.fx_graph_path, repo, public_safe, &mut stats)?;
            normalize_field(&mut graph.inductor_ir_path, repo, public_safe, &mut stats)?;
        }

        // ── kernel paths + sources (§2.1 / §2.4) ──
        for kernel in artifact.kernels.iter_mut() {
            normalize_field(&mut kernel.source_path, repo, public_safe, &mut stats)?;
            // PTX and kernel source are model IP — always omitted under a share-safe level.
            if kernel.ptx_path.take().is_some() {
                stats.kernel_sources_removed += 1;
            }
            if kernel.kernel_source_excerpt.take().is_some() {
                stats.kernel_sources_removed += 1;
            }
        }

        // ── lint finding source locations (§2.1 / §2.3) ──
        for finding in artifact.lint_findings.iter_mut() {
            if let Some(loc) = finding.source_location.as_mut() {
                normalize_field(&mut loc.file, repo, public_safe, &mut stats)?;
                if loc.code_excerpt.take().is_some() {
                    stats.source_excerpts_removed += 1;
                }
            }
        }
    }

    artifact.session.redaction_policy = target;
    Ok(stats)
}

/// Normalize one optional path field in place, bumping `stats.paths_normalized` when the value
/// actually changed. Every path-bearing field in the artifact (graph paths, kernel source paths,
/// lint file locations) routes through this one helper so they all get the same rule and the same
/// `CLS-E0008` credential-dir refusal, rather than re-deriving it per call site.
fn normalize_field(
    field: &mut Option<String>,
    repo: Option<&str>,
    public_safe: bool,
    stats: &mut RedactionStats,
) -> Result<(), ClsError> {
    if let Some(path) = field.as_ref() {
        let normalized = rules::normalize_path(path, repo, public_safe)?;
        if &normalized != path {
            stats.paths_normalized += 1;
        }
        *field = Some(normalized);
    }
    Ok(())
}

/// One category of the share-safety audit (`cl scrub --verify`): a category name, whether it
/// passed, and a human detail line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyCheck {
    pub passed: bool,
    pub message: String,
}

/// The result of auditing an artifact against a `target` redaction level — a per-category list of
/// [`VerifyCheck`]s plus the artifact's own declared policy. The artifact is **share-safe for
/// `target`** exactly when every check passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub declared: RedactionPolicy,
    pub target: RedactionPolicy,
    pub checks: Vec<VerifyCheck>,
}

impl VerifyReport {
    /// Whether the artifact already meets every share-safety guarantee of `target`.
    pub fn is_clean(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }
}

/// Audit whether `artifact` already satisfies the redaction guarantees of `target` — the read-only
/// counterpart to [`redact_artifact`]. It never mutates; it reports, per category, whether a leak
/// remains (an unscrubbed token, a raw host, an un-normalized path, a present source/kernel
/// excerpt, a non-allowlisted env var). The categories mirror the rules [`redact_artifact`]
/// applies, so "verify is clean" means "scrubbing to `target` would change nothing".
///
/// A non-strict `target` (`internal` / `confidential`) makes no share-safety promise, so the audit
/// is vacuously clean.
pub fn verify(artifact: &ClsArtifact, target: RedactionPolicy) -> VerifyReport {
    let declared = artifact.session.redaction_policy;
    let mut checks = Vec::new();

    if is_strict(target) {
        let public_safe = matches!(target, RedactionPolicy::PublicSafe);
        let detected_repo = rules::detect_repo();
        let repo = detected_repo.as_deref();

        // ── command ──
        match artifact.session.command.as_ref() {
            Some(_) if public_safe => checks.push(VerifyCheck {
                passed: false,
                message: "command is present (public-safe nulls it)".into(),
            }),
            Some(cmd) => {
                let has_token = rules::scrub_command(cmd, false).1;
                checks.push(VerifyCheck {
                    passed: !has_token,
                    message: if has_token {
                        "command contains an unscrubbed secret-like token".into()
                    } else {
                        "no unscrubbed argv tokens".into()
                    },
                });
            }
            None => checks.push(VerifyCheck {
                passed: true,
                message: "no command recorded".into(),
            }),
        }

        // ── host ──
        match artifact.session.host.as_ref() {
            Some(_) if public_safe => checks.push(VerifyCheck {
                passed: false,
                message: "host is present (public-safe nulls it)".into(),
            }),
            Some(h) if !rules::is_hashed_host(h) => checks.push(VerifyCheck {
                passed: false,
                message: "host is not FQDN-hashed".into(),
            }),
            _ => checks.push(VerifyCheck {
                passed: true,
                message: if public_safe {
                    "host is nulled".into()
                } else {
                    "host is FQDN-hashed".into()
                },
            }),
        }

        // ── env vars ──
        let leaky_env: Vec<&String> = artifact
            .session
            .env_snapshot
            .as_ref()
            .map(|env| {
                env.relevant_env_vars
                    .keys()
                    .filter(|k| !rules::env_var_allowed(k))
                    .collect()
            })
            .unwrap_or_default();
        checks.push(VerifyCheck {
            passed: leaky_env.is_empty(),
            message: if leaky_env.is_empty() {
                "env vars are allowlisted".into()
            } else {
                format!("{} non-allowlisted env var(s) present", leaky_env.len())
            },
        });

        // ── paths (graph / kernel / lint) ──
        let mut path_leaks = 0usize;
        let mut credential_dir = false;
        let mut check_path = |p: &Option<String>| {
            if let Some(path) = p {
                match rules::normalize_path(path, repo, public_safe) {
                    Ok(normalized) if &normalized != path => path_leaks += 1,
                    Ok(_) => {}
                    Err(_) => credential_dir = true, // CLS-E0008: a credential-dir path is present
                }
            }
        };
        for g in &artifact.compiled_graphs {
            check_path(&g.fx_graph_path);
            check_path(&g.inductor_ir_path);
        }
        for k in &artifact.kernels {
            check_path(&k.source_path);
        }
        for f in &artifact.lint_findings {
            if let Some(loc) = &f.source_location {
                check_path(&loc.file);
            }
        }
        let path_ok = path_leaks == 0 && !credential_dir;
        checks.push(VerifyCheck {
            passed: path_ok,
            message: if credential_dir {
                "a path under a credential directory is present".into()
            } else if path_leaks > 0 {
                format!("{path_leaks} un-normalized path(s) present")
            } else {
                "no raw paths present".into()
            },
        });

        // ── source excerpts (lint) ──
        let source_excerpts = artifact
            .lint_findings
            .iter()
            .filter(|f| {
                f.source_location
                    .as_ref()
                    .is_some_and(|l| l.code_excerpt.is_some())
            })
            .count();
        checks.push(VerifyCheck {
            passed: source_excerpts == 0,
            message: if source_excerpts == 0 {
                "no source excerpts present".into()
            } else {
                format!("{source_excerpts} source excerpt(s) present")
            },
        });

        // ── kernel sources (PTX / excerpt) ──
        let kernel_sources = artifact
            .kernels
            .iter()
            .filter(|k| k.ptx_path.is_some() || k.kernel_source_excerpt.is_some())
            .count();
        checks.push(VerifyCheck {
            passed: kernel_sources == 0,
            message: if kernel_sources == 0 {
                "no kernel sources present".into()
            } else {
                format!("{kernel_sources} kernel source(s) present")
            },
        });
    }

    VerifyReport {
        declared,
        target,
        checks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small but realistic artifact carrying a leak in every sensitive field, parsed from
    /// JSON so the test exercises the real on-disk shape rather than a hand-built struct.
    fn leaky_artifact(policy: &str) -> ClsArtifact {
        let json = format!(
            r#"{{
              "schema_version": "0.5.0",
              "session": {{
                "id": "00000000-0000-4000-8000-000000000000",
                "timestamp": "2026-06-17T00:00:00Z",
                "torch_version": "2.11.0",
                "redaction_policy": "{policy}",
                "host": "ml-prod-07.megacorp.internal",
                "command": "python /home/jdoe/work/megacorp-llm/train.py --hf-token=hf_abcdefghijklmnopqrst",
                "env_snapshot": {{
                  "relevant_env_vars": {{
                    "TORCH_LOGS": "+recompiles",
                    "HF_TOKEN": "hf_secret",
                    "PATH": "/usr/bin"
                  }},
                  "random_seed": 0
                }}
              }},
              "kernels": [{{
                "kernel_id": "k0",
                "name": "fused_attn",
                "source_path": "/home/jdoe/work/megacorp-llm/kern.py",
                "ptx_path": "/tmp/abc.ptx",
                "kernel_source_excerpt": "secret triton source"
              }}],
              "lint_findings": [{{
                "finding_id": "f0",
                "pattern_category": "in_place_on_alias",
                "severity": "high",
                "source_location": {{
                  "file": "/home/jdoe/work/megacorp-llm/model.py",
                  "line_start": 47,
                  "code_excerpt": "x.relu_()"
                }}
              }}]
            }}"#
        );
        serde_json::from_str(&json).expect("test artifact is valid")
    }

    #[test]
    fn demote_is_refused_with_e0009() {
        let mut artifact = leaky_artifact("default-strict");
        let err =
            redact_artifact(&mut artifact, RedactionPolicy::Internal, "salt", false).unwrap_err();
        assert_eq!(err.code(), "CLS-E0009");
        // The artifact is untouched on refusal.
        assert_eq!(
            artifact.session.redaction_policy,
            RedactionPolicy::DefaultStrict
        );
    }

    #[test]
    fn promote_to_default_strict_scrubs_every_field() {
        let mut artifact = leaky_artifact("confidential");
        let stats =
            redact_artifact(&mut artifact, RedactionPolicy::DefaultStrict, "salt", false).unwrap();

        let s = &artifact.session;
        assert_eq!(s.redaction_policy, RedactionPolicy::DefaultStrict);
        // host hashed, not raw.
        assert_eq!(
            s.host.as_deref().unwrap(),
            rules::hash_host("ml-prod-07.megacorp.internal", "salt")
        );
        // argv token scrubbed, flag preserved.
        let cmd = s.command.as_deref().unwrap();
        assert!(cmd.contains("--hf-token=<scrubbed>"));
        assert!(!cmd.contains("hf_abcdefghijklmnopqrst"));
        // env: secret dropped, torch var kept, non-torch var dropped.
        let env = &s.env_snapshot.as_ref().unwrap().relevant_env_vars;
        assert!(env.contains_key("TORCH_LOGS"));
        assert!(!env.contains_key("HF_TOKEN"));
        assert!(!env.contains_key("PATH"));
        // kernel IP gone.
        let k = &artifact.kernels[0];
        assert!(k.ptx_path.is_none());
        assert!(k.kernel_source_excerpt.is_none());
        // lint excerpt gone.
        assert!(artifact.lint_findings[0]
            .source_location
            .as_ref()
            .unwrap()
            .code_excerpt
            .is_none());

        assert!(stats.host_redacted);
        assert!(stats.argv_tokens_scrubbed);
        assert_eq!(stats.env_vars_dropped, 2);
        assert_eq!(stats.kernel_sources_removed, 2);
        assert_eq!(stats.source_excerpts_removed, 1);
    }

    #[test]
    fn promote_to_public_safe_nulls_host_and_command() {
        let mut artifact = leaky_artifact("default-strict");
        redact_artifact(&mut artifact, RedactionPolicy::PublicSafe, "salt", false).unwrap();
        assert_eq!(
            artifact.session.redaction_policy,
            RedactionPolicy::PublicSafe
        );
        assert!(
            artifact.session.host.is_none(),
            "public-safe nulls the host"
        );
        assert!(
            artifact.session.command.is_none(),
            "public-safe nulls the command"
        );
    }

    #[test]
    fn promote_to_public_safe_normalizes_compiled_graph_paths() {
        // A compiled-graph path is path-bearing too (matches the Python collector, which
        // normalizes fx_graph_path / inductor_ir_path). public-safe's {home}/ catch-all
        // anonymizes it even when repo detection can't match the basename.
        let json = r#"{
          "schema_version": "0.5.0",
          "session": {
            "id": "00000000-0000-4000-8000-000000000000",
            "timestamp": "2026-06-17T00:00:00Z", "torch_version": "2.11.0",
            "redaction_policy": "confidential"
          },
          "compiled_graphs": [{
            "graph_id": "g0",
            "fx_graph_path": "/home/jdoe/secret-project/fx.txt",
            "inductor_ir_path": "/home/jdoe/secret-project/ir.txt"
          }]
        }"#;
        let mut artifact: ClsArtifact = serde_json::from_str(json).unwrap();
        redact_artifact(&mut artifact, RedactionPolicy::PublicSafe, "salt", false).unwrap();
        let g = &artifact.compiled_graphs[0];
        assert_eq!(
            g.fx_graph_path.as_deref().unwrap(),
            "{home}/secret-project/fx.txt"
        );
        assert_eq!(
            g.inductor_ir_path.as_deref().unwrap(),
            "{home}/secret-project/ir.txt"
        );
        assert!(!g.fx_graph_path.as_deref().unwrap().contains("jdoe"));
    }

    #[test]
    fn promote_to_same_level_is_idempotent() {
        let mut once = leaky_artifact("confidential");
        redact_artifact(&mut once, RedactionPolicy::DefaultStrict, "salt", false).unwrap();
        let mut twice = once.clone();
        // Re-scrubbing an already-default-strict artifact must be a fixed point.
        redact_artifact(&mut twice, RedactionPolicy::DefaultStrict, "salt", false).unwrap();
        assert_eq!(once, twice, "scrub is idempotent");
    }

    #[test]
    fn strictness_orders_levels_for_promote_only() {
        assert!(strictness(RedactionPolicy::Confidential) < strictness(RedactionPolicy::Internal));
        assert!(strictness(RedactionPolicy::Internal) < strictness(RedactionPolicy::DefaultStrict));
        assert!(
            strictness(RedactionPolicy::DefaultStrict) < strictness(RedactionPolicy::PublicSafe)
        );
        assert!(is_strict(RedactionPolicy::DefaultStrict));
        assert!(is_strict(RedactionPolicy::PublicSafe));
        assert!(!is_strict(RedactionPolicy::Internal));
        assert!(!is_strict(RedactionPolicy::Confidential));
    }

    #[test]
    fn verify_flags_a_leaky_artifact() {
        let artifact = leaky_artifact("confidential");
        let report = verify(&artifact, RedactionPolicy::DefaultStrict);
        assert!(
            !report.is_clean(),
            "a raw confidential artifact is not share-safe"
        );
        // The host, command, env, and kernel-source categories should each fail.
        let failed: Vec<&str> = report
            .checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.message.as_str())
            .collect();
        assert!(failed.iter().any(|m| m.contains("host")));
        assert!(failed.iter().any(|m| m.contains("token")));
        assert!(failed.iter().any(|m| m.contains("env")));
        assert!(failed.iter().any(|m| m.contains("kernel source")));
    }

    #[test]
    fn verify_passes_after_scrubbing() {
        let mut artifact = leaky_artifact("confidential");
        redact_artifact(&mut artifact, RedactionPolicy::DefaultStrict, "salt", false).unwrap();
        let report = verify(&artifact, RedactionPolicy::DefaultStrict);
        assert!(
            report.is_clean(),
            "a default-strict-scrubbed artifact verifies clean; failures: {:?}",
            report
                .checks
                .iter()
                .filter(|c| !c.passed)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn verify_default_strict_is_not_public_safe() {
        let mut artifact = leaky_artifact("confidential");
        redact_artifact(&mut artifact, RedactionPolicy::DefaultStrict, "salt", false).unwrap();
        // Clean for default-strict, but public-safe demands host+command nulled.
        let report = verify(&artifact, RedactionPolicy::PublicSafe);
        assert!(!report.is_clean());
        assert!(report
            .checks
            .iter()
            .any(|c| !c.passed && c.message.contains("host is present")));
    }
}
