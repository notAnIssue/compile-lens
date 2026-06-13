# ADR-036: DB-driven granular detection for Tool 4

- **Status**: Accepted
- **Date**: 2026-06-13
- **Deciders**: project maintainer
- **Related**: ADR-013 (four mandatory items per pattern); ADR-032 (two-layer lint); ADR-035 (TOML database format); the `cls-correctness-db` crate and the Layer-1 scanner.

## Context

Tool 4's analyzer matches a scanner candidate to the correctness database by
`lookup(pattern_category)` — an exact match on the pattern's `name`. The Layer-1 scanner, however,
emitted only two fixed categories: `in_place_op_on_alias` (structural) and a single generic
`operator_non_default_param`. So a database grown to the taxonomy's 30+ distinctly-named patterns
would have most of its entries **unmatchable** — no scanner output would ever carry their category.
The number of patterns the database can *hold* had become decoupled from the number of categories
the scanner can *emit*, making "30+ patterns" a hollow target.

The structural pattern is fine: one built-in detector, one category, many underlying issues. The
gap is the operator family — the taxonomy's "single operator with non-default parameters" class
(`diag_embed` with a negative `dim1`, and so on), where each operator is its own bug with its own
issue and workaround, but the scanner lumped them under one category.

## Decision

Make detection **database-driven and granular**. An operator-family pattern carries a `[detector]`
naming the operator and the parameters whose explicit use flags a candidate:

```toml
[patterns.detector]
operator = "diag_embed"
params = ["dim1", "dim2", "offset"]
```

The Python front-end reads these detectors to configure the scanner, which — on a match — emits a
candidate whose `pattern_category` is **that pattern's own `name`**. The analyzer's
`lookup(pattern_category)` then lands on exactly that entry for its evidence. The two halves meet on
the pattern `name`: the scanner emits it, the analyzer looks it up. Structural patterns
(`in_place_op_on_alias`) keep their built-in detector and carry no `[detector]`.

The database is therefore the single source of truth for both halves of a pattern: the Python side
reads the *detection rule*, the Rust side reads the *evidence and severity*. Adding a pattern is a
data edit — a new entry with a detector and the four ADR-013 items — and detection extends
automatically, with no code change.

Layer-1 stays static AST: it sees that a watched parameter was *passed*, not its value, so the
detector is a deliberately coarse proxy for "non-default." That is correct for a *candidate* — the
database join, Layer-2 graph confirmation, and a human refine it — and the negative fixture guards
against the over-flag widening.

## Consequences

- **Positive**: the 30+ catalogue is matchable — each operator pattern emits its own category and
  resolves to its own evidence; the structural pattern is unchanged. Growing the catalogue is data
  entry, not code. The database is one editable source feeding detection and evidence alike.
- **Costs**: the scanner's output vocabulary is now defined by the database names — an intentional
  coupling (the database is the contract). `--db` must be passed to *both* halves — the Python scan
  (to build detection rules) and the Rust analysis (for evidence); the single-command front-end
  that passes it once is the Phase 7 hero.
- **Follow-ups**: this change ships the schema (`detector`), the granular-emission scanner, and a
  seed database proving the mechanism with verified patterns (one per detection mode). The
  front-end wiring that reads detectors and drives the scanner, and the bulk of the catalogue, land
  next.

## Alternatives considered

| Criterion (weight) | A: granular category = name | B: generic category + analyzer reverse-match | C: per-pattern hardcoded detectors |
|---|---|---|---|
| Match-path simplicity (3) | 9 (27) | 5 (15) | 8 (24) |
| Add-pattern-as-data / one source (3) | 9 (27) | 9 (27) | 3 (9) |
| Scanner/analyzer decoupling (2) | 6 (12) | 8 (16) | 4 (8) |
| Implementation churn now (2) | 8 (16) | 5 (10) | 3 (6) |
| **Total** | **8.2** | **6.8** | **4.7** |

- **B (generic category, analyzer reverse-matches by detector)** keeps the scanner pattern-agnostic
  — appealing — but moves matching into the analyzer (find the entry whose detector matches the
  candidate's recorded op/param), needs the candidate to carry that op/param, and must resolve one
  operator watched by several patterns. The complexity lands on every match instead of once at
  emission.
- **C (a hardcoded detector function per pattern)** abandons the data-file model: 30+ patterns
  become 30+ pieces of detection code, and adding one needs a toolchain and a recompile. It would
  win only if zero database dependency were paramount, which ADR-035 already decided against.
