//! `cls-correctness-db` — the known-buggy-pattern database for Tool 4 (`compile-lint`).
//!
//! Tool 4 flags `torch.compile` anti-patterns. For a finding to be credible rather than a guess,
//! every pattern it can flag must carry the **four mandatory items of ADR-013**: a minimal repro, a
//! real PyTorch issue link, a workaround, and a positive + negative test fixture. Those four are
//! **required fields**, so a pattern entry missing any of them fails to load (serde reports the
//! missing field) — the discipline is enforced by the type, not by a convention reviewers must
//! remember.
//!
//! The database is authored in **TOML** (ADR-035): a stable, mature dependency, with literal
//! multi-line strings (`'''…'''`) that hold the Python code blocks verbatim. It is an internal,
//! maintainer-curated artifact — end users run the CLI and read findings; they never edit this file.
//!
//! This crate is the schema, the loader, and the four-item enforcement. The 30+ production patterns
//! land later.

use serde::{Deserialize, Serialize};

/// Positive and negative test fixtures for a pattern (ADR-013 item 4). Both are required — the
/// negative case guards the pattern against false positives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixtures {
    /// Code the pattern *should* match.
    pub positive: String,
    /// Similar code the pattern should *not* match.
    pub negative: String,
}

/// One known-buggy `torch.compile` anti-pattern.
///
/// The four ADR-013 items — `minimal_repro`, `pytorch_issue`, `workaround`, `fixtures` — are
/// required; everything else is optional metadata that refines severity or attribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pattern {
    /// Stable identifier, e.g. `in_place_op_on_alias`.
    pub name: String,
    /// ADR-013 item 1 — a minimal reproducer demonstrating the bug.
    pub minimal_repro: String,
    /// ADR-013 item 2 — a real PyTorch issue URL (evidence this is a reported bug, not a guess).
    pub pytorch_issue: String,
    /// ADR-013 item 3 — a code-level workaround.
    pub workaround: String,
    /// ADR-013 item 4 — positive + negative test fixtures.
    pub fixtures: Fixtures,

    /// Li et al. 2026 taxonomy section, if the pattern maps to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub li_et_al_section: Option<String>,
    /// PyTorch version that fixed the issue, if known (drives severity downgrade).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_in_version: Option<String>,
    /// Optional base severity hint (`low` / `medium` / `high`); the analyzer's escalation rules
    /// have the final say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,
}

/// The pattern database: a flat list of patterns, in authored (file) order.
///
/// Load it from TOML with [`PatternDb::from_toml`]; an entry missing any ADR-013 item fails there.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PatternDb {
    #[serde(default)]
    patterns: Vec<Pattern>,
}

impl PatternDb {
    /// Load and validate a database from TOML. A pattern missing any of the four mandatory ADR-013
    /// items fails here — the error names the missing field.
    pub fn from_toml(toml_src: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_src)
    }

    /// Look up a pattern by name. Linear over the curated set (tens of entries), and deterministic
    /// because the set keeps its authored order.
    pub fn lookup(&self, name: &str) -> Option<&Pattern> {
        self.patterns.iter().find(|p| p.name == name)
    }

    /// All patterns, in authored order.
    pub fn patterns(&self) -> &[Pattern] {
        &self.patterns
    }

    /// Number of patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A complete entry with all four ADR-013 items (and some optional metadata).
    const COMPLETE: &str = r#"
[[patterns]]
name = "in_place_op_on_alias"
minimal_repro = '''
y = x.expand(2, 3)
y[0] = 1   # in-place write through an alias
'''
pytorch_issue = "https://github.com/pytorch/pytorch/issues/114302"
workaround = "y = x.expand(2, 3).clone()"
li_et_al_section = "§3.2.3"
fixed_in_version = "2.5"

[patterns.fixtures]
positive = "y = x.expand(2, 3); y[0] = 1"
negative = "y = x.expand(2, 3).clone(); y[0] = 1"
"#;

    #[test]
    fn loads_a_complete_pattern_and_looks_it_up() {
        let db = PatternDb::from_toml(COMPLETE).expect("complete entry loads");
        assert_eq!(db.len(), 1);
        let p = db.lookup("in_place_op_on_alias").expect("found by name");
        assert_eq!(
            p.pytorch_issue,
            "https://github.com/pytorch/pytorch/issues/114302"
        );
        assert!(p.minimal_repro.contains("in-place write"));
        assert_eq!(p.fixtures.negative, "y = x.expand(2, 3).clone(); y[0] = 1");
        assert_eq!(p.li_et_al_section.as_deref(), Some("§3.2.3"));
    }

    #[test]
    fn missing_a_mandatory_item_fails_to_load_naming_the_field() {
        // Same entry, workaround removed — the loader must reject it (ADR-013 enforcement).
        let no_workaround = r#"
[[patterns]]
name = "p"
minimal_repro = "x"
pytorch_issue = "https://github.com/pytorch/pytorch/issues/1"
[patterns.fixtures]
positive = "a"
negative = "b"
"#;
        let err = PatternDb::from_toml(no_workaround).expect_err("must reject");
        assert!(
            err.to_string().contains("workaround"),
            "error should name the missing field: {err}"
        );
    }

    #[test]
    fn missing_a_fixture_case_fails_to_load() {
        // The negative fixture is required (false-positive guard); omitting it is rejected.
        let no_negative = r#"
[[patterns]]
name = "p"
minimal_repro = "x"
pytorch_issue = "https://github.com/pytorch/pytorch/issues/1"
workaround = "w"
[patterns.fixtures]
positive = "a"
"#;
        let err = PatternDb::from_toml(no_negative).expect_err("must reject");
        assert!(
            err.to_string().contains("negative"),
            "error should name the missing fixture: {err}"
        );
    }

    #[test]
    fn optional_metadata_may_be_absent() {
        let minimal = r#"
[[patterns]]
name = "p"
minimal_repro = "x"
pytorch_issue = "https://github.com/pytorch/pytorch/issues/1"
workaround = "w"
[patterns.fixtures]
positive = "a"
negative = "b"
"#;
        let db = PatternDb::from_toml(minimal).expect("loads without optional metadata");
        let p = db.lookup("p").unwrap();
        assert!(p.li_et_al_section.is_none());
        assert!(p.severity.is_none());
    }

    #[test]
    fn lookup_miss_returns_none_and_empty_db_loads() {
        let db = PatternDb::from_toml(COMPLETE).unwrap();
        assert!(db.lookup("nonexistent").is_none());

        let empty = PatternDb::from_toml("").expect("empty db is valid");
        assert!(empty.is_empty());
    }
}
