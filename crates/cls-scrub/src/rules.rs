//! The pure redaction primitives — the share-time mirror of the collect-time Python
//! redactor (`python/compile_lens/security/redactor.py`).
//!
//! These functions are the building blocks the policy layer ([`crate::redact_artifact`]) calls
//! field by field. They are deliberately a 1:1 port of the Python primitives so a
//! `default-strict` artifact written by the collector and the same artifact re-scrubbed here
//! land on byte-identical output — the two halves of the toolkit speak one redaction
//! vocabulary (discipline D8), and the scrub regression corpus pins that agreement.
//!
//! Spec: `docs/06_security/redaction_policy.md` §2.

use std::sync::LazyLock;

use cls_errors::ClsError;
use regex::Regex;
use sha2::{Digest, Sha256};

/// A path containing any of these segments is refused outright (never recorded / never
/// emitted), because everything under it is a credential. Mirrors the Python tuple.
const SENSITIVE_DIRS: [&str; 3] = ["/.ssh/", "/.aws/", "/.gnupg/"];

// ── command (argv) token scrubbing — §2.2 ────────────────────────────────────────────────
/// `(pattern, replacement)` applied in order; each keeps the flag/key and redacts the value.
/// `${1}` re-emits the first capture group (the `--flag=` prefix) so only the secret is lost.
static COMMAND_SCRUBS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"(--api[_-]?key=)\S+", "${1}<scrubbed>"),
        (r"(--hf-token=)\S+", "${1}<scrubbed>"),
        (r"(--wandb-key=)\S+", "${1}<scrubbed>"),
        (r"(--token=)\S+", "${1}<scrubbed>"),
        (r"(--password=)\S+", "${1}<scrubbed>"),
        (r"(--secret=)\S+", "${1}<scrubbed>"),
        (r"(--auth=)\S+", "${1}<scrubbed>"),
        (r"\bBearer\s+[A-Za-z0-9._-]+", "Bearer <scrubbed>"),
        (r"\bhf_[A-Za-z0-9]{20,}", "hf_<scrubbed>"),
        (r"\bsk-[A-Za-z0-9]{20,}", "sk-<scrubbed>"),
        (r"\bAKIA[A-Z0-9]{16}", "AKIA<scrubbed>"),
        (r"\bxox[abopr]-[A-Za-z0-9-]+", "xox<scrubbed>"),
    ]
    .into_iter()
    .map(|(p, r)| {
        (
            Regex::new(p).expect("static command-scrub pattern is valid"),
            r,
        )
    })
    .collect()
});

/// Extra strict-mode patterns (§2.2): long base64-looking or hex-looking runs that could be a
/// novel secret no named pattern caught. Applied only when the target demands it (public-safe
/// or an explicit `--strict`), because they can over-match a legitimate long argument.
static STRICT_TOKEN_SCRUBS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"\b[A-Za-z0-9+/]{32,}\b", "<scrubbed:long-string>"),
        (r"\b[A-Fa-f0-9]{32,}\b", "<scrubbed:long-hex>"),
    ]
    .into_iter()
    .map(|(p, r)| {
        (
            Regex::new(p).expect("static strict-scrub pattern is valid"),
            r,
        )
    })
    .collect()
});

/// Redact secret-looking tokens from an argv string, preserving each flag/key.
///
/// `strict` additionally scrubs long base64/hex runs (§2.2). Returns the scrubbed string and
/// whether any replacement fired (the caller uses the bool for the operator-facing count).
pub fn scrub_command(command: &str, strict: bool) -> (String, bool) {
    let mut out = command.to_string();
    let mut changed = false;
    for (pattern, replacement) in COMMAND_SCRUBS.iter() {
        if pattern.is_match(&out) {
            out = pattern.replace_all(&out, *replacement).into_owned();
            changed = true;
        }
    }
    if strict {
        for (pattern, replacement) in STRICT_TOKEN_SCRUBS.iter() {
            if pattern.is_match(&out) {
                out = pattern.replace_all(&out, *replacement).into_owned();
                changed = true;
            }
        }
    }
    (out, changed)
}

// ── path normalization — §2.1 ─────────────────────────────────────────────────────────────
const SITE_PACKAGES: &str = r"/lib/python[\d.]+/site-packages";

/// Install/cache path rules applied in order; first match wins. None of these need the repo
/// basename — they collapse known PyTorch/Triton install and cache trees.
static PATH_RULES: LazyLock<Vec<(Regex, String)>> = LazyLock::new(|| {
    [
        (
            format!(r"^/opt/conda/envs/[^/]+{SITE_PACKAGES}/torch/(.*)$"),
            r"{torch_install}/${1}".to_string(),
        ),
        (
            format!(r"^/opt/conda/envs/[^/]+{SITE_PACKAGES}/triton/(.*)$"),
            r"{triton_install}/${1}".to_string(),
        ),
        (
            format!(r"^/usr/local{SITE_PACKAGES}/torch/(.*)$"),
            r"{torch_install}/${1}".to_string(),
        ),
        (
            r"^.*\.cache/torch_extensions/(.*)$".to_string(),
            r"{torch_cache}/${1}".to_string(),
        ),
        (
            r"^.*\.cache/torch_compile/(.*)$".to_string(),
            r"{torch_compile_cache}/${1}".to_string(),
        ),
    ]
    .into_iter()
    .map(|(p, r)| (Regex::new(&p).expect("static path rule is valid"), r))
    .collect()
});

/// Catch-all home-directory anonymizer used only when targeting `public-safe`: any leftover
/// `/home/<user>/…` or `/Users/<user>/…` prefix becomes `{home}/…` so no username survives.
static HOME_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^/(?:home|Users)/[^/]+/(.*)$").expect("static home rule is valid")
});

/// Collapse a filesystem path to a non-identifying placeholder form.
///
/// Returns [`ClsError::SensitivePathDetected`] (`CLS-E0008`) if the path is inside a credential
/// directory — such a path is never emitted, even redacted. When `repo` is provided, a
/// `/home/<user>/…/<repo>/<rest>` (and the macOS `/Users` variant) collapses to `{repo}/<rest>`.
/// When `public_safe` is set, any remaining home-directory prefix is anonymized to `{home}/`.
/// Unrecognized absolute paths are returned unchanged (`cl scrub --verify` flags those).
pub fn normalize_path(
    path: &str,
    repo: Option<&str>,
    public_safe: bool,
) -> Result<String, ClsError> {
    if SENSITIVE_DIRS.iter().any(|d| path.contains(d)) {
        return Err(ClsError::SensitivePathDetected {
            path: path.to_string(),
        });
    }

    if let Some(repo) = repo {
        let repo_rule = Regex::new(&format!(
            r"^/(?:home|Users)/[^/]+/(?:.+/)?{}/(.*)$",
            regex::escape(repo)
        ))
        .expect("repo rule built from escaped basename is valid");
        if let Some(caps) = repo_rule.captures(path) {
            return Ok(format!("{{repo}}/{}", &caps[1]));
        }
    }

    for (pattern, replacement) in PATH_RULES.iter() {
        let new = pattern.replace(path, replacement.as_str());
        if new != path {
            return Ok(new.into_owned());
        }
    }

    if public_safe {
        if let Some(caps) = HOME_PREFIX.captures(path) {
            return Ok(format!("{{home}}/{}", &caps[1]));
        }
    }

    Ok(path.to_string())
}

// ── host FQDN hashing — §2.5 ──────────────────────────────────────────────────────────────
/// Stable per-machine identifier `dh-<8hex>` = `sha256(fqdn + salt)[:8]`.
///
/// The salt is a per-install secret, so the hash is stable on one machine but does not leak the
/// FQDN and is not correlatable across machines. Identical to the Python `hash_host`.
pub fn hash_host(fqdn: &str, salt: &str) -> String {
    let digest = Sha256::digest(format!("{fqdn}{salt}").as_bytes());
    format!("dh-{}", &hex::encode(digest)[..8])
}

/// Whether a host value is *already* an FQDN hash (`dh-<8 lowercase hex>`).
///
/// Re-scrubbing an artifact must be a fixed point (the corpus pins idempotence), so a host that
/// was hashed on a previous scrub must not be hashed again — hashing the hash would drift on
/// every pass. This is the guard that keeps the host field stable across repeated scrubs.
pub fn is_hashed_host(host: &str) -> bool {
    host.strip_prefix("dh-")
        .is_some_and(|rest| rest.len() == 8 && rest.bytes().all(|b| b.is_ascii_hexdigit()))
}

// ── repo detection — §2.1 ─────────────────────────────────────────────────────────────────
/// Find the git repo basename by walking up from the current directory looking for a `.git`
/// entry, mirroring the Python collector's repo detection. Used to collapse
/// `/home/<user>/…/<repo>/<rest>` paths to `{repo}/<rest>`.
///
/// Returns `None` when no `.git` is found on the way up (e.g. scrubbing an artifact far from
/// where it was collected). That is the safe direction: an unrecognized `/home/<user>/` path is
/// then left intact under default-strict (and `cl scrub --verify` flags it) and anonymized to
/// `{home}/` under public-safe, so a missing repo basename never causes a username to leak at
/// the public-safe level.
pub fn detect_repo() -> Option<String> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join(".git").exists() {
            return dir.file_name()?.to_str().map(str::to_string);
        }
        if !dir.pop() {
            return None;
        }
    }
}

// ── env var filtering — §2.6 ──────────────────────────────────────────────────────────────
const ENV_WHITELIST_PREFIXES: [&str; 7] = [
    "TORCH_",
    "TORCHINDUCTOR_",
    "TORCHDYNAMO_",
    "TRITON_",
    "CUDA_",
    "CUDNN_",
    "HSA_",
];
const ENV_WHITELIST_EXACT: [&str; 1] = ["NCCL_DEBUG"];
const ENV_DENY_SUFFIXES: [&str; 6] = [
    "_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_AUTH",
    "_CREDENTIAL",
];
const ENV_DENY_EXACT: [&str; 4] = [
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
];

/// Decide whether a single env var name survives the allowlist/denylist.
///
/// The denylist wins over the allowlist (`TORCH_API_KEY` is dropped), so a secret can never be
/// captured by widening the allowlist — fail-closed, matching Python `filter_env_vars`.
pub fn env_var_allowed(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    if ENV_DENY_EXACT.contains(&upper.as_str())
        || ENV_DENY_SUFFIXES.iter().any(|s| upper.ends_with(s))
    {
        return false;
    }
    ENV_WHITELIST_EXACT.contains(&upper.as_str())
        || ENV_WHITELIST_PREFIXES.iter().any(|p| upper.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_command_redacts_value_keeps_flag() {
        let (out, changed) = scrub_command(
            "python train.py --hf-token=hf_abcdefghijklmnopqrstuvwxyz --lr 0.001",
            false,
        );
        assert!(changed);
        assert_eq!(
            out, "python train.py --hf-token=<scrubbed> --lr 0.001",
            "the flag survives, only the token is redacted"
        );
    }

    #[test]
    fn scrub_command_catches_bare_provider_tokens() {
        let (out, _) = scrub_command(
            "run --model sk-0123456789abcdefghijABCDEF Bearer ya29.abcDEF-_token",
            false,
        );
        assert!(out.contains("sk-<scrubbed>"));
        assert!(out.contains("Bearer <scrubbed>"));
        assert!(!out.contains("0123456789abcdefghij"));
    }

    #[test]
    fn scrub_command_no_secret_is_a_noop() {
        let (out, changed) = scrub_command("python train.py --epochs 3", false);
        assert!(!changed);
        assert_eq!(out, "python train.py --epochs 3");
    }

    #[test]
    fn strict_mode_scrubs_long_unrecognized_runs() {
        // A 40-char base64-ish blob no named pattern catches.
        let blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNA";
        let plain = format!("python deploy.py --mystery {blob}");
        let (lenient, _) = scrub_command(&plain, false);
        assert!(lenient.contains(blob), "lenient mode leaves it alone");
        let (strict, changed) = scrub_command(&plain, true);
        assert!(changed);
        assert!(!strict.contains(blob), "strict mode catches the novel blob");
    }

    #[test]
    fn normalize_path_collapses_repo_and_install_trees() {
        assert_eq!(
            normalize_path(
                "/home/jdoe/work/megacorp-llm/model.py",
                Some("megacorp-llm"),
                false
            )
            .unwrap(),
            "{repo}/model.py",
        );
        assert_eq!(
            normalize_path(
                "/opt/conda/envs/prod/lib/python3.11/site-packages/torch/_inductor/x.py",
                None,
                false,
            )
            .unwrap(),
            "{torch_install}/_inductor/x.py",
        );
    }

    #[test]
    fn normalize_path_public_safe_anonymizes_leftover_home() {
        // No repo known, but public-safe must not leak the username.
        assert_eq!(
            normalize_path("/home/jdoe/scratch/notebook.py", None, true).unwrap(),
            "{home}/scratch/notebook.py",
        );
        // Without public_safe the same path is left intact (verify flags it instead).
        assert_eq!(
            normalize_path("/home/jdoe/scratch/notebook.py", None, false).unwrap(),
            "/home/jdoe/scratch/notebook.py",
        );
    }

    #[test]
    fn normalize_path_refuses_credential_dirs() {
        let err = normalize_path("/home/jdoe/.ssh/id_rsa", None, false).unwrap_err();
        assert_eq!(err.code(), "CLS-E0008");
    }

    #[test]
    fn hash_host_is_stable_and_does_not_leak_fqdn() {
        let h1 = hash_host("ml-prod-07.megacorp.internal", "salt-a");
        let h2 = hash_host("ml-prod-07.megacorp.internal", "salt-a");
        assert_eq!(h1, h2, "same fqdn+salt is stable");
        assert!(h1.starts_with("dh-"));
        assert_eq!(h1.len(), 3 + 8);
        assert!(!h1.contains("megacorp"));
        assert_ne!(
            h1,
            hash_host("ml-prod-07.megacorp.internal", "salt-b"),
            "a different install salt yields a different hash (not cross-machine correlatable)"
        );
    }

    #[test]
    fn env_denylist_wins_over_allowlist() {
        assert!(env_var_allowed("TORCH_LOGS"));
        assert!(env_var_allowed("TRITON_CACHE_DIR"));
        assert!(env_var_allowed("NCCL_DEBUG"));
        // Denylist wins even though the prefix is allowlisted.
        assert!(!env_var_allowed("TORCH_API_KEY"));
        assert!(!env_var_allowed("HF_TOKEN"));
        assert!(!env_var_allowed("OPENAI_API_KEY"));
        // Not on the allowlist at all.
        assert!(!env_var_allowed("PATH"));
    }
}
