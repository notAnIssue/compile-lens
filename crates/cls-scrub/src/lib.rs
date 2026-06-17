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

        // ── kernel paths + sources (§2.1 / §2.4) ──
        for kernel in artifact.kernels.iter_mut() {
            if let Some(path) = kernel.source_path.as_ref() {
                let normalized = rules::normalize_path(path, repo, public_safe)?;
                if &normalized != path {
                    stats.paths_normalized += 1;
                }
                kernel.source_path = Some(normalized);
            }
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
                if let Some(file) = loc.file.as_ref() {
                    let normalized = rules::normalize_path(file, repo, public_safe)?;
                    if &normalized != file {
                        stats.paths_normalized += 1;
                    }
                    loc.file = Some(normalized);
                }
                if loc.code_excerpt.take().is_some() {
                    stats.source_excerpts_removed += 1;
                }
            }
        }
    }

    artifact.session.redaction_policy = target;
    Ok(stats)
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
}
