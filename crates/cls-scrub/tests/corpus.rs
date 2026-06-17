//! Scrub regression corpus + fuzz — the release-blocker gate for `cls-scrub`.
//!
//! Three guarantees, pinned so a regression is a red CI check that unambiguously means "do not
//! release a leaky scrub":
//!
//! 1. **Byte-identical golden files**: every `test_corpus/before/*` file, scrubbed, matches its
//!    committed `after/*` counterpart byte for byte (deterministic output — D10).
//! 2. **The scrubbed output is share-safe**: each scrubbed artifact passes [`cls_scrub::verify`]
//!    against the level it was scrubbed to, and a re-scrub is a fixed point (idempotent).
//! 3. **Fuzz**: a secret value behind a recognized flag never survives the scrub, for thousands of
//!    random values, via a deterministic PRNG (no `rand` dep, so a failure reproduces exactly).
//!
//! The JSON corpus is scrubbed to **public-safe** specifically because that level's path rules are
//! repo-detection-independent (the `{home}/` catch-all), so the golden files are identical on any
//! machine — `default-strict` path normalization depends on the cwd's git repo and would not be
//! reproducible in CI.
//!
//! Regenerate the golden `after/` files after an intended rule change: `CORPUS_BLESS=1 cargo test
//! -p cls-scrub --test corpus`, then review the diff before committing.

use std::path::PathBuf;

use cls_schema::{ClsArtifact, RedactionPolicy};

/// A fixed salt so the FQDN host hash (`dh-<…>`) is deterministic in the golden files.
const SALT: &str = "corpus-fixed-salt";

const JSON_FIXTURES: &[&str] = &[
    "session-leaks-paths.cls.json",
    "session-leaks-tokens.cls.json",
    "session-leaks-kernel-source.cls.json",
];

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_corpus")
}

fn blessing() -> bool {
    std::env::var_os("CORPUS_BLESS").is_some()
}

/// Scrub one JSON fixture to public-safe and either write the golden file (bless) or assert it
/// matches byte-for-byte and is itself share-safe.
fn check_json(name: &str) {
    let dir = corpus_dir();
    let before = std::fs::read_to_string(dir.join("before").join(name)).expect("read before");
    let mut artifact: ClsArtifact = serde_json::from_str(&before).expect("before parses");
    cls_scrub::redact_artifact(&mut artifact, RedactionPolicy::PublicSafe, SALT, false)
        .expect("scrub succeeds");
    let scrubbed = serde_json::to_string(&artifact).expect("serialize");

    let after_path = dir.join("after").join(name);
    if blessing() {
        // Newline-terminated so the golden file satisfies the repo's end-of-file-fixer hook.
        std::fs::write(&after_path, format!("{scrubbed}\n")).expect("write golden");
        return;
    }

    let expected = std::fs::read_to_string(&after_path).expect("read golden after/ file");
    assert_eq!(
        scrubbed,
        expected.trim_end_matches('\n'),
        "{name}: scrub output drifted from the committed after/ golden file"
    );
    // The scrubbed artifact must actually be share-safe for the level it was scrubbed to.
    let reparsed: ClsArtifact = serde_json::from_str(&scrubbed).expect("after parses");
    assert!(
        cls_scrub::verify(&reparsed, RedactionPolicy::PublicSafe).is_clean(),
        "{name}: the scrubbed artifact does not verify clean for public-safe"
    );
}

#[test]
fn corpus_leaks_paths() {
    check_json("session-leaks-paths.cls.json");
}

#[test]
fn corpus_leaks_tokens() {
    check_json("session-leaks-tokens.cls.json");
}

#[test]
fn corpus_leaks_kernel_source() {
    check_json("session-leaks-kernel-source.cls.json");
}

#[test]
fn corpus_html_xss_vectors() {
    let dir = corpus_dir();
    let name = "report-xss-vectors.html";
    let before = std::fs::read_to_string(dir.join("before").join(name)).expect("read before");
    let (scrubbed, _) = cls_scrub::html::scrub_html(&before);

    let after_path = dir.join("after").join(name);
    if blessing() {
        std::fs::write(
            &after_path,
            format!("{}\n", scrubbed.trim_end_matches('\n')),
        )
        .expect("write golden");
        return;
    }
    let expected = std::fs::read_to_string(&after_path).expect("read golden after/ file");
    assert_eq!(
        scrubbed.trim_end_matches('\n'),
        expected.trim_end_matches('\n'),
        "{name}: html scrub drifted from golden"
    );
    // Re-scrubbing the cleaned report is a fixed point.
    let (twice, stats) = cls_scrub::html::scrub_html(&scrubbed);
    assert!(
        stats.is_noop() && twice == scrubbed,
        "html scrub not idempotent"
    );
}

#[test]
fn corpus_scrub_is_idempotent() {
    if blessing() {
        return;
    }
    for name in JSON_FIXTURES {
        let before =
            std::fs::read_to_string(corpus_dir().join("before").join(name)).expect("read before");
        let mut once: ClsArtifact = serde_json::from_str(&before).expect("parse");
        cls_scrub::redact_artifact(&mut once, RedactionPolicy::PublicSafe, SALT, false).unwrap();
        let mut twice = once.clone();
        cls_scrub::redact_artifact(&mut twice, RedactionPolicy::PublicSafe, SALT, false).unwrap();
        assert_eq!(once, twice, "{name}: scrub(scrub(x)) != scrub(x)");
    }
}

#[test]
fn fuzz_flagged_secret_value_never_survives() {
    // A secret value behind a recognized flag is scrubbed regardless of its content. We feed 4096
    // random alphanumeric values (each a single `\S+` token) through a deterministic LCG — no
    // `rand` dependency, so any failure reproduces from the same seed.
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.";
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    for _ in 0..4096 {
        let len = 8 + (next() % 40) as usize;
        let value: String = (0..len)
            .map(|_| ALPHABET[(next() as usize) % ALPHABET.len()] as char)
            .collect();
        let command = format!("python serve.py --api-key={value} --port 8080");
        let (scrubbed, changed) = cls_scrub::rules::scrub_command(&command, false);
        assert!(changed, "a flagged secret must register as changed");
        assert!(
            !scrubbed.contains(&value),
            "secret value survived the scrub: {value} -> {scrubbed}"
        );
        assert!(
            scrubbed.contains("--api-key=<scrubbed>"),
            "the flag must be preserved: {scrubbed}"
        );
    }
}
