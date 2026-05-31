# Contributing

`compile-lens` is pre-alpha. External contributions are not being accepted yet
(see the README's *Contributing* note for the reason and the timeline). This
document captures the working rules the maintainers themselves follow, so the
public state of the repository stays consistent with them.

If you have found a **security issue**, follow [`SECURITY.md`](./SECURITY.md) —
do not open a public issue or PR.

## Naming

Every artifact that travels with this repository — PR titles, commit messages,
branches, code comments, docs, ADRs — refers to a unit of work by an
identifier that **`the design doc` or an ADR already establishes**:

- A **Tool** by its number and name from `the design doc` §3 (*Tool 1 — recompile aggregator*).
- A **crate** by its cargo name (`cls-schema`, `cls-errors`, `cls-cli`, …).
- An **engineering discipline** by its design-doc tag (*D7 algorithm*, *D8 error UX*, …, *D11 security*).
- An **architectural decision** by its ADR number (*ADR-021*, *ADR-022*, *ADR-027*).

Identifiers that come from private planning notes (section IDs of the form
`S<phase>.<num>`, planning PR codes, paths under the maintainer's planning
tree) **must not appear** in repository artifacts. A pre-commit hook
(`scripts/check_private_fingerprints.sh`) refuses to commit a file that
contains them; `scripts/preflight.sh` re-checks the whole tree before push.

The full rationale, including alternatives that were rejected, is
[ADR-028](./docs/02_design_decisions/adr-028-naming-and-delivery.md).

### Branches

Branch names are intermediate artifacts, not durable references, but the
naming convention is the same:

- `feat/<crate-or-feature>` — a new capability.
- `fix/<short-slug>` — a defect fix.
- `refactor/<area>` — internal rearrangement.
- `chore/<area>` — release engineering, infrastructure, dependencies.
- `docs/<area>` — documentation-only.

Use the cargo name or the design-doc identifier in the slug (`feat/tool-1-recompile-summary`, `fix/cls-errors-help-formatting`).

## Local checks

Before pushing, run one command:

```bash
./scripts/preflight.sh
```

It runs every check CI runs: `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo test --workspace`, `ruff check`, `ruff format --check`, `mypy`, the
Python unit + integration tests, every `pre-commit` hook, and a
private-fingerprint sweep across the whole tree.

`--quick` skips the integration tests for a faster inner loop:

```bash
./scripts/preflight.sh --quick
```

If the script is green, the PR should land green. If CI fails afterward, the
discrepancy is itself a bug — fix it in `scripts/preflight.sh` or in the
workflow, not in the next PR's verification habit. **CI is a gate, not a
debugger.**

## Bundling units of work into PRs

Implementation work breaks down into ~1–2 hour units. The default is **one
unit per PR**. A PR may bundle multiple units only when *one* of these holds:

1. They live in the same crate or subsystem **and** share a test target that
   can't be split (e.g. a Cargo manifest + the first module that depends on
   the manifest's metadata).
2. There is a sequential dependency that prevents independent testing
   (e.g. introducing a CLI flag and the subcommand that consumes it).
3. The whole bundle is Tier C boilerplate — templates, config files, license
   headers, ignore files — where the cost of separate PRs exceeds the review
   benefit.

Any other reason to bundle goes in the PR description.

## Changelog

Every PR that has user-visible effect adds at least one line under
`[Unreleased]` in `CHANGELOG.md`. Phase ship rolls `[Unreleased]` into a
versioned section, so per-PR entries accumulate naturally and there is no
phase-end changelog sweep.

PRs that are purely internal (test refactors, dependency bumps already covered
by Dependabot, comment fixes) may skip the changelog entry — the PR template's
checklist asks the author to make that judgement.

## ADRs

Open an ADR when the decision is **hard to reverse**: architectural shape,
schema-level commitments, public-API choices, project-wide process changes.
ADRs live in `docs/02_design_decisions/`, numbered sequentially, named
`adr-<NNN>-<short-slug>.md`. The template is at
[`_template.md`](./docs/02_design_decisions/_template.md).

Reserve nothing. ADR-026 was reserved during planning and never filled,
which is why ADR-027 skipped the gap. Numbers are assigned at the moment a
decision is made, not in advance.

Decisions with two or more viable options use a **weighted decision matrix**
in the *Alternatives considered* section: criteria with weights summing to 10,
every option scored 0–10 on the same scale, weighted total per option, and a
sentence on which weight or score would flip the result. Don't move scores to
get the answer you want — move the answer.

## ADR / commit / PR consistency

A change that is in scope for an ADR should land in the same PR that ships
its first implementation. A reader following the ADR back to the code should
land in the commit that introduced both — not in an out-of-date ADR that
references unrelated commit ranges.
