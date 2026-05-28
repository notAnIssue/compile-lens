# ADR-022: Error handling — `thiserror` for the enum, `miette` for diagnostics

- **Status**: Accepted
- **Date**: 2026-05-28
- **Deciders**: project maintainer
- **Related**: implemented in `crates/cls-errors`; consumed by `crates/cls-cli` and the
  analyzer crates. The decision originated in `development_workflow.md §2` (worked example);
  this ADR is the formal record.

## Context

design.md discipline **D8** requires every user-facing error to render five parts: a
stable error code (`CLS-EXXXX`), an optional source span, a cause chain, an actionable
hint, and a documentation link. Rust's `std::error::Error` provides almost none of this —
at best a `Display` message and a `source()` chain. The project needs a typed error enum
that supports stable codes (for testing, doc lookup, CI greps) **and** renders rich
diagnostics for CLI and IDE consumers.

The constraint is mild but real: stable codes are part of the public contract — users will
google `CLS-E0001` — and the renderer must work without a TTY (CI logs) as well as with one
(developer terminals).

## Decision

`ClsError` is a single project-wide enum declared with **both** `#[derive(thiserror::Error)]`
and `#[derive(miette::Diagnostic)]`. Each variant:

- gets its `Display` message and `#[source]` cause chain from `thiserror`;
- gets its stable `CLS-EXXXX` code and an actionable `help` string from miette via
  `#[diagnostic(code(...), help(...))]`.

An inherent `ClsError::code(&self) -> &'static str` returns the stable code via an
exhaustive match — independent of `miette::Diagnostic::code`. A registry test (see
`crates/cls-errors/tests/registry.rs`) pins the inherent match arms and the derive
attributes to agree, so a future divergence fails CI rather than silently shipping.

`cls-cli` returns `miette::Result<()>` from `main`, so any `ClsError` propagated with `?`
is rendered by miette's `GraphicalReportHandler` (the `fancy` feature) — colored output,
indented help, and the cause chain.

### Doc URL rendering is deferred (intentional)

The fifth D8 part — a doc URL per code — is **not committed in this ADR**. There is no
public docs site yet, and we explicitly do not want to bake a specific URL or domain
(e.g. `compile-lens.dev`) into ten variants when the hosting decision has not been made.

The chosen `thiserror + miette` mechanism still supports URLs out of the box — adding
`url(...)` back to each variant (or via a single shared helper) is a one-line change per
variant when a docs site exists, with **no application-code impact**. Until then the
canonical doc page for each top-five code lives in-repo at `docs/08_errors/<code>.md`.

## Consequences

- **Positive**:
  - All five D8 parts *achievable* with one derive each — no custom `Display`/`Debug`
    code. Four of the five (code, cause chain, hint, optional span) ship today; URL is
    deferred (see above) and arrives with a one-line per-variant attribute when ready.
  - Stable `CLS-EXXXX` codes are part of the enum's source — easy to grep, easy to test,
    easy to enumerate for the docs page.
  - Cause chains preserve the underlying `io::Error` / `serde_json::Error` etc. via
    `#[source]`, so call sites get the full failure context.
- **Negative / costs**:
  - Two derive macros plus miette's `fancy` feature pull in extra deps (`thiserror`,
    `miette`, `owo-colors`, `supports-color`, `unicode-width`). The added compile time
    and binary size are accepted for the diagnostic UX.
  - A new variant must touch the enum, the inherent `code()` match, and (for the
    top-five user-facing codes) a doc page. The registry test enforces the first two.
  - miette 7's `Diagnostic` derive expands code that trips the `unused_assignments` lint
    on each bound field; the crate uses a single targeted `#![allow(unused_assignments)]`,
    scoped to this crate so real assignment bugs elsewhere stay visible.
- **Follow-ups**:
  - The top-five user-facing codes (`E0001`, `E0002`, `E0007`, `E0008`, `E0009`) get full
    `docs/08_errors/<code>.md` pages; the remaining five are auto-listed in
    `docs/08_errors/all.md` with one-line summaries.
  - When a downstream crate (e.g. `cls-schema-migrate`) needs to fail with a typed error,
    it returns `ClsError` directly rather than introducing a parallel enum.

## Alternatives considered

- **A — `anyhow` only**: ergonomic for application code but no stable codes, no typed
  matching, no structured rendering. Loses the D8 contract.
- **B — `thiserror` only**: typed enum, can carry a code string in the `Display` message
  by hand, but no help / doc URL / colored rendering. Loses most of D8.
- **C — `miette` only**: miette can derive both `Error` and `Diagnostic` itself, replacing
  thiserror. Viable alternative to D; loses the well-known "thiserror for Error, miette
  for Diagnostic" idiom that most Rust reviewers expect.
- **D — `thiserror` + `miette`** (chosen).

### Weighted decision matrix

Criteria weighted to sum to 10; every option scored 0–10 on the same scale; weighted
contribution = `weight × (raw/10)`.

| Criterion (weight) — why this weight | A anyhow | B thiserror | C miette | D both ✅ |
|---|---|---|---|---|
| **Stable error codes (3.0)** — D8 mandates them; users grep / google these | 2 (0.60) | 7 (2.10) | 9 (2.70) | **10 (3.00)** |
| **Diagnostic rendering — five D8 parts (2.5)** — colored hint, doc URL, optional span | 3 (0.75) | 4 (1.00) | 10 (2.50) | **10 (2.50)** |
| **Type safety / exhaustive match (1.5)** — every new variant handled at call sites | 3 (0.45) | 10 (1.50) | 9 (1.35) | **10 (1.50)** |
| **Cause chain via `#[source]` (1.5)** — preserve underlying errors across layers | 8 (1.20) | 10 (1.50) | 9 (1.35) | **10 (1.50)** |
| **Ecosystem idiom / maintainability (1.5)** — recognizable to Rust reviewers | 8 (1.20) | 9 (1.35) | 7 (1.05) | **10 (1.50)** |
| **Weighted total / 10** | **4.20** | **7.45** | **8.95** | **10.00** |

**Readout**: D wins decisively, dominating every criterion. C is the close runner-up
(miette alone can replicate most of D's surface); the deciding factor is the
"thiserror + miette" combo's idiom recognition and the clean split (thiserror = `Error`
trait, miette = `Diagnostic` protocol). The result would only soften toward B if a
downstream constraint forbade miette's deps, in which case B would be the fallback at
the cost of UX.
