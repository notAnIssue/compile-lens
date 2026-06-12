# ADR-035: TOML for the correctness-DB pattern format

- **Status**: Accepted
- **Date**: 2026-06-12
- **Deciders**: project maintainer
- **Related**: ADR-013 (the four mandatory items per pattern); ADR-032 (Tool 4 two-layer lint); `the design doc` §8.4 (Tool 4); the `cls-correctness-db` crate.

## Context

Tool 4 (`compile-lint`) flags `torch.compile` anti-patterns against a database of known-buggy
patterns. ADR-013 requires every pattern to carry four items — a minimal repro, a real PyTorch
issue link, a workaround, and a positive + negative test fixture — so each finding cites evidence
rather than a guess. Three of those items are **multi-line code**, so the database needs a
human-authored format with first-class multi-line strings, loaded and validated by the
`cls-correctness-db` crate (an entry missing any item must fail to load).

The design originally said YAML. But the Rust YAML ecosystem is currently in a poor state:
`serde_yaml` (the de-facto standard) was **archived/unmaintained in 2024**, which this project's
weekly security audit flags as a RUSTSEC advisory; the maintained fork `serde_yml` is only at a
`0.0.x` release and already churns its minimum supported Rust version. Pulling either into a
conservative, security-gated workspace is a real cost.

A pseudo-criterion to name and discard: *"YAML is more familiar to the PyTorch audience."* The
pattern database is a **maintainer-curated, internal artifact** — end users run the CLI, read the
findings (which inline the repro / workaround / issue), and suppress via in-code comments; they
never open this file. And a future *contributor* of a pattern is by definition a power user (they
must cite a real PyTorch issue and write a minimal repro) who already authors `pyproject.toml`
daily — TOML does not block them. So format familiarity carries no real weight here.

## Decision

Author the correctness DB in **TOML**, loaded by the `toml` crate (a stable 1.x dependency that
Cargo itself uses). TOML literal multi-line strings (`'''…'''`) hold the Python code blocks verbatim
with no escaping. The four ADR-013 items are **required serde fields** on the `Pattern` /
`Fixtures` types, so an entry missing any of them fails `from_toml`, naming the missing field — the
discipline is enforced by the type, not by reviewer vigilance.

## Consequences

- **Positive**: a stable, actively-maintained dependency that passes the security audit cleanly;
  literal multi-line strings ideal for code; the database stays a reviewable, independently-editable
  data file (friendlier to a future contributor than embedding patterns in Rust).
- **Negative / costs**: deviates from the design's stated YAML — a deliberate, execution-time format
  choice, not an architectural change. One wart: a repro whose code literally contains `'''` would
  collide with the TOML delimiter; minimal repros essentially never do, and `"""…"""` is the escape
  hatch.
- **Follow-ups**: the 30+ production patterns are authored against this schema later; the analyzer
  joins scanner output with this DB and is oblivious to the on-disk format.

## Alternatives considered

| Criterion (weight) | A: TOML | B: serde_yml 0.0.x | C: serde_yaml + audit-ignore | D: Rust-native |
|---|---|---|---|---|
| Dependency health / clean audit (4) | 9 (3.6) | 4 (1.6) | 3 (1.2) | 10 (4.0) |
| Four-item enforcement (2) | 8 (1.6) | 8 (1.6) | 8 (1.6) | 10 (2.0) |
| Stays a reviewable data file (2) | 7 (1.4) | 9 (1.8) | 9 (1.8) | 4 (0.8) |
| Implementation simplicity (1.5) | 8 (1.2) | 8 (1.2) | 7 (1.05) | 5 (0.75) |
| Contributor ergonomics (0.5) | 8 (0.4) | 8 (0.4) | 8 (0.4) | 4 (0.2) |
| **Total** | **8.2** | **6.6** | **6.05** | **7.75** |

- **B (serde_yml)** sinks on dependency health: a `0.0.x` fork that already churns its Rust-version
  floor, pulled in only to match the literal word "YAML".
- **C (serde_yaml + audit-ignore)** is worst: knowingly depending on an unmaintained crate *and*
  suppressing the advisory works against the security gate's purpose.
- **D (Rust-native patterns)** is the close runner-up (zero dependency, compile-time enforcement);
  it loses only on "stays a data file" — patterns would live in Rust, needing a toolchain and a
  recompile to edit. The result would flip to D only if "zero dependencies at all costs" outweighed
  keeping an editable data file; it would flip to a YAML option only if end users (not power-user
  contributors) were expected to read or edit the database directly, which they are not.
