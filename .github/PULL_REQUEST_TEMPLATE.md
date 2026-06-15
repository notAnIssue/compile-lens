<!--
Thanks for opening a PR! Please fill the three sections below.
Mechanical box-ticking is not the goal — the engineering checklist mirrors the
project's engineering disciplines (D7–D11). Reviewer
judgement is what they actually pattern-match against; the checkboxes just
make sure each axis was at least considered.
-->

## What does this PR do?

<!-- One short paragraph: the change and its purpose. Skip diff stats — GitHub shows those. -->

## Engineering Checklist (D7–D11)

- [ ] **D7 — Algorithm**: any algorithm change states its confidence, complexity, and failure modes (or "N/A — no algorithm touched").
- [ ] **D8 — Error UX**: new error variants live in `cls-errors` with a stable `code`, an actionable `help`, and a doc page if they make the top-10 user-facing list.
- [ ] **D9 — Observability**: new code paths carry `#[tracing::instrument]` (Rust) / `structlog` context (Python) so failures are diagnosable from production logs.
- [ ] **D10 — Migration**: any schema change has a paired `cls-schema-migrate` step and keeps reproducibility fields intact (pre-V1 exception: detect-and-refuse is enough).
- [ ] **D11 — Security**: new collector fields have a sensible `redaction_policy` default and a scrub-regression test.
- [ ] **CI**: new tests / new gated behavior land in CI (`.github/workflows/ci.yml`) — not a follow-up PR.
- [ ] **API stability**: public API breaks carry a deprecation note in the changelog; semver is respected.
- [ ] **Performance**: hot-path or algorithmic changes are covered by a benchmark (or explicitly justified as cold-path).

## Changelog

<!--
Every PR with user-visible effect adds at least one line under [Unreleased]
in CHANGELOG.md so the per-phase ship is a rename, not a sweep. Purely
internal changes (test refactors, comment fixes) may skip — make the call.
See CONTRIBUTING.md "Changelog" for the rule.
-->

- [ ] `CHANGELOG.md` updated under `[Unreleased]`, or this PR is internal-only.

## Related ADR / Issue

<!--
- Closes #123
- ADR-NNN: docs/02_design_decisions/adr-NNN-*.md
- Design context inline below; do not link to non-public planning notes.
-->
