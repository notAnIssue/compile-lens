# ADR-028: One public naming namespace; private identifiers CI-gated

- **Status**: Accepted
- **Date**: 2026-05-31
- **Deciders**: project maintainer
- **Related**: ADR-021 (schema layout), ADR-022 (error handling), ADR-027 (unknown-field capture); the engineering disciplines (D7–D11).

## Context

Phase 0 ran with **four parallel identifier schemes** for the same work:

1. **Phase numbers** (`Phase 0`, `Phase 1`, …) — pacing concept.
2. **Section IDs** (`S0.4`, `S0.16`, …) — 1–2 hour planning units inside a phase.
3. **PR codes** (`P1`, `P2`, …, `P9`) — delivery batches.
4. **Feature names** (`cls-errors`, `cls-schema-migrate`, `tracing setup`, …) — what shipped.

Of those four, only (4) appears in design documents that travel with the
repository. (1)–(3) live in the maintainer's planning tree, which contributors
to the public repository cannot read.

The schemes did not stay separated. `S<phase>.<num>` identifiers and absolute
paths into the planning tree leaked into PR descriptions, commit messages, code
comments, ADR text, and Python / Rust docstrings — 13 occurrences of section
IDs plus 9 occurrences of planning paths, requiring two retroactive scrub PRs
(#13 and #19) to clean. A reader of the public repository, encountering
`S0.10`, has no way to resolve it.

**Pseudo-criterion, rejected:** "harmonize the names by renaming sections after
features (e.g. `S0.6` → `schema-rust-bindings`)." This treats the leak as a
naming-aesthetics problem. It is not — the next category of private identifier
(phase code-names, internal milestones, future planning labels) leaks in
exactly the same way, and a naming convention does not prevent it. The leak
needs a mechanical gate, not a vocabulary change.

## Decision

**One identifier namespace is allowed in artifacts that travel with this
repository, and it is the namespace the public design docs and the ADR series
already establish.** Specifically:

- **Tools** by name and number — *Tool 1 (recompile aggregator)*, *Tool 2 (compile-diff)*, …
- **Crates** by their cargo name — `cls-schema`, `cls-errors`, `cls-cli`, …
- **Engineering disciplines** by their design-doc tag — *D7 algorithm*, *D8 error UX*, … *D11 security*.
- **Architectural decisions** by ADR number — *ADR-021*, *ADR-022*, *ADR-027*.

**Identifiers from private planning documents are gated out of this
repository.** Specifically:

- Section IDs of the form `S<phase>.<num>` (and the placeholder forms `S0.X`, `S0.N`).
- PR codes `P<n>`, phase numbers, and other planning-internal labels.
- Absolute paths under the maintainer's planning tree (`colibri/`, `assets/notes/…`, …).

The gate is mechanical: `scripts/check_private_fingerprints.sh` runs as a
pre-commit hook and as part of `scripts/preflight.sh`. Files that exist *to
describe the rule* (the hook itself, `CONTRIBUTING.md`, this ADR) are
whitelisted by exact path.

**Planning-internal identifiers continue to exist.** They are the right
granularity for the maintainer's day-to-day work, and the rule does not ask
them to go away. The rule says only: they do not cross into artifacts a
contributor will read.

## Consequences

- **Positive**.
  - Any identifier in a repository artifact resolves against documents that
    travel with the repository. A reader needs no out-of-band context.
  - The recurring "did anything leak?" sweep is replaced by a `git commit`
    that fails when the patterns appear. The class of bug is closed.
  - Phase boundaries become invisible to repository readers — which is
    correct, because phases are an internal pacing concept, not a contract.
- **Negative / costs**.
  - Identifiers in artifacts are longer ("the `cls-schema-migrate` crate"
    instead of "S0.17"). The cost is at write time, not read time, and writing
    is rarer than reading.
  - One more pre-commit hook to debug if it false-positives. The whitelist
    is by exact path and grows by one entry per ADR / contributor doc that
    needs to name the pattern.
- **Follow-ups**.
  - `CONTRIBUTING.md` captures the practical naming + delivery rules.
  - `scripts/preflight.sh` consolidates every local gate behind one command.
  - `.github/PULL_REQUEST_TEMPLATE.md` includes a per-PR changelog prompt
    so `CHANGELOG.md` is maintained as work lands, not in a phase-end sweep.
  - CI workflows gain `concurrency` so a re-push cancels the previous run.

## Alternatives considered

- **Option A — Rename sections to be self-describing names** (e.g.
  `schema-rust-bindings` instead of `S0.6`). Keeps section granularity, makes
  any leak readable as English. But it solves leakage by vocabulary rather than
  by gate, and the next category of private identifier leaks the same way.
- **Option B — Drop section-level planning; plan and track by PR only.** Removes
  the leak surface entirely. But Phase 0 had nine PRs across ~60 hours, an
  average of ~7 hours per PR — too coarse for "what should I do this morning?"
  planning. The granularity gap reappears as internal scratch notes that drift
  out of sync with the PR list.
- **Option C — Keep section-level planning private; gate private identifiers
  out of public artifacts; reuse the public design docs' existing namespace for
  public reference.** Chosen.

### Weighted decision matrix

Criteria weighted to sum to 10; every option scored 0–10 on the same scale;
weighted contribution = `weight × (raw/10)`.

| Criterion (weight) — why this weight | A rename | B PRs-only | C gated reuse ✅ |
|---|---|---|---|
| **Day-to-day planning granularity (2.5)** — 1–2 hour units are the minimum useful for "what should I do next?" | 9 (2.25) | 4 (1.00) | **9 (2.25)** |
| **Leak resistance against the next private-fingerprint category (2.5)** — leakage was the trigger; the fix must generalize | 6 (1.50) | 9 (2.25) | **10 (2.50)** |
| **Self-describing to an outside reader (2.0)** — identifiers in repo artifacts should resolve without out-of-band context | 8 (1.60) | 8 (1.60) | **9 (1.80)** |
| **Cognitive simplicity (1.5)** — number of concurrent name systems the maintainer holds | 7 (1.05) | 9 (1.35) | **8 (1.20)** |
| **Reversibility (1.5)** — can we walk this back if it turns out wrong | 6 (0.90) | 5 (0.75) | **9 (1.35)** |
| **Weighted total / 10** | **7.30** | **6.95** | **9.10** |

**Readout**: C dominates the two highest-weighted axes — planning granularity
*and* leak resistance — without paying the granularity cost of B. A is the
runner-up; it scores well on granularity and cognitive simplicity but loses on
leak resistance, because a naming convention does not stop the *next* category
of private identifier from leaking, while a mechanical gate does. The result
would flip to A only if "leak resistance" were re-weighted below 1.5 — but the
two retroactive scrub PRs already done are direct evidence that the weight is
at least the value above.
