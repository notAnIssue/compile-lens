# Tool 4 — `compile-lint`

Flag source-level anti-patterns that are known to correlate with `torch.compile` **correctness**
divergence — silent wrong results, not crashes. Each finding cites a real PyTorch issue, a minimal
repro, and a workaround from the correctness database, so it is evidence, not a guess.

`compile-lint` is a **lint, not an oracle**: it recognizes a bounded set of *statically detectable*
anti-patterns and says nothing about the rest. A clean report does not mean the model is
compile-safe — it means none of the catalogued, statically-recognizable patterns were found. See
*Part C* for exactly what is and isn't covered.

## Part A — Theory & reference

### What it does

Two layers (ADR-032), both pure static analysis — the user's code is parsed, never executed:

- **Layer 1 — AST scan.** Parses source with the standard `ast` module and emits *candidates*. Safe
  to run in CI on untrusted source. Two detection modes (ADR-036):
  - **structural** — recognized by built-in structure, e.g. `in_place_op_on_alias` (an in-place
    write through a tensor that aliases another, created by `view` / `expand` / `unfold` / …).
  - **operator-family** — driven by the correctness database: a pattern names an operator and the
    parameters whose explicit use flags a candidate, and the scanner emits that pattern's name as
    the candidate's category.
- **Layer 2 — functionalized-graph confirmation.** For input-mutation patterns, confirms a
  candidate against the AOTAutograd functionalized graph (needs the live function + example inputs,
  so it is a programmatic step, not part of the file scan).

### The database is the single source of truth

The correctness database (TOML) carries, per pattern, the four mandatory items (ADR-013): a minimal
repro, a real PyTorch issue URL, a workaround, and positive + negative fixtures. It feeds both
halves of the tool (ADR-036): the Python front-end reads each pattern's detector rule to configure
the scanner, and the Rust analyzer reads the evidence and assigns severity. They meet on the pattern
name. Adding a pattern is a data edit; detection extends with no code change.

### Severity is torch-version aware

A pattern whose upstream fix landed at or below the user's torch version is downgraded `high → info`
(the bug no longer bites them); an unknown or higher fix version stays `high` (the conservative
side). A surviving `high` makes `cl compile-lint` exit 1 so it can gate CI.

## Part B — Examples

The Python front-end scans source into a `.cls.json`; the Rust analyzer joins it with the database.
(Unifying these into one command is a later milestone; today they are two steps.)

```bash
# 1. Scan a file or a directory — the database's detector rules drive operator detection.
cl compile-lint path/to/model.py --db data/correctness_db.toml -o lint.cls.json

# 2. Analyze: join candidates with the database, render, exit 1 if any `high` survives.
cl compile-lint lint.cls.json --db data/correctness_db.toml --format sarif > lint.sarif
```

Without `--db`, only the structural pattern fires (no operator rules). Findings can be suppressed
in source with `# compile-lint: ignore[pattern]`, a `# compile-lint: file-ignore[pattern]` comment,
or a `@compile_lint_ignore("pattern")` decorator.

## Part C — Detection coverage & limitations

This is the most important section: a lint that overstates its reach is worse than none. Tool 4
covers the **statically detectable** slice of the `torch.compile` correctness-bug taxonomy (Li et
al.) and explicitly does not cover the rest.

| Bug family (taxonomy) | Detectable by Tool 4? | Why |
|---|---|---|
| In-place op on a tensor alias (§3.2.3) | ✅ **structural** | The alias creation + in-place write is visible in the AST. |
| Operator with a non-default parameter (§3.2.2) | ✅ **operator-family** | A watched operator called with a watched keyword parameter is visible in the AST. |
| Optimization-triggering operator *sequences* (§3.2.2, e.g. `split`+`stack`, `addmm`+`cat`) | ⚠️ **not yet** | Would need an AST sequence detector (a future detection mode). |
| Graph semantic capturing — custom logic, execution-context mutation (§3.2.1) | ❌ no | Needs to model tracing / runtime context, not visible statically. |
| Graph caching — stale cached graph across mutating state (§3.2.1) | ❌ no | Only surfaces under *repeated execution* with changing state. |
| Memory layout conflicts (§3.2.3) | ❌ no | Needs layout/stride metadata tracking, not in source. |
| Low-level codegen — numerical / extreme values (§3.2.2) | ❌ no | Triggered by input *values*, not by source shape. |

Consequences of being a static, candidate-producing lint:

- **Coarse "non-default" proxy.** Layer 1 sees that a parameter was *passed*, not its value. So
  `diag_embed(dim1=-1)` and `diag_embed(dim1=1)` both flag, though only the negative dim is buggy.
  This is correct for a *candidate* — the database (is this a known bug?), Layer 2, and a human
  refine it. The negative fixture in each pattern guards the over-flag from widening.
- **Keyword arguments only.** A bug triggered by a *positional* argument (e.g. an operator whose
  buggy parameter is conventionally positional) is not flagged; the watched parameter must be passed
  by keyword.
- **The catalogue is small and honest.** It holds the patterns that are both statically detectable
  and backed by a cited issue — currently a handful, growing as more are verified. It is **not** a
  count target: the taxonomy's ~116 bugs are mostly in the ❌ rows above, reachable only by Layer 2
  or runtime differential testing, not by a static scan. We would rather ship a small set of real,
  evidence-backed patterns than a large set of guesses.

## Related

- `divergence` (Tool 3) — when a model *does* diverge under compile, localize and attribute it at
  runtime. Complementary: `compile-lint` is the cheap static pre-filter; `divergence` is the
  runtime investigation.
- ADR-032 (two-layer lint), ADR-035 (database format), ADR-036 (DB-driven granular detection).
