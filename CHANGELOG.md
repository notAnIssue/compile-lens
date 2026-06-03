# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `compile_lens.collectors.recompile.RecompileCollector` — the Tool 1
  collector core. Accumulates `Recompilation` / `CompiledGraph` records via
  an additive `add_records` pipeline, dispatches over the three input modes
  (`collect(mode, source)`; an unknown mode fails closed with `ValueError`),
  and `finalize()` writes a schema-valid `.cls.json`. The per-mode parsers
  (Mode A logs / Mode B tlparse / Mode C dynamo.explain) and command
  scrubbing land in later sections; this PR establishes the surface they
  plug into. The redaction policy is coerced through the fail-closed
  `RedactionPolicy` enum and recorded in the session.
- `CONTRIBUTING.md` capturing the naming, bundling, changelog, and ADR rules
  that the maintainers follow (ADR-028 holds the rationale).
- `scripts/preflight.sh` — one command runs every CI check locally; `--quick`
  skips the integration tests for a faster inner loop.
- `scripts/check_private_fingerprints.sh` — pre-commit hook that refuses
  commits containing identifiers from internal planning notes (see ADR-028
  for the exact pattern families).
- ADR-028 — *one naming namespace, anchored in `the design doc`; private
  identifiers CI-gated*. Closes the leak class addressed retroactively by
  the `v0.5.0-alpha.0` cleanup PRs.

### Changed

- CI workflows now use `concurrency:` — a re-push to the same PR branch
  cancels the in-flight run, saving Actions minutes during iteration. Pushes
  to `main` still run to completion.
- `.github/PULL_REQUEST_TEMPLATE.md` carries a per-PR changelog checkbox so
  `CHANGELOG.md` is maintained as work lands.
- README's *Development setup* section points at `CONTRIBUTING.md` and the
  preflight script instead of listing hooks inline.
- A pre-push hook (`scripts/check_branch_advances.sh`) refuses any push
  whose branch tip equals `main`. This catches the failure mode where a
  pre-commit hook silently aborts a commit but its non-zero exit code is
  swallowed by a `| tail` pipe — the subsequent push uploads the unchanged
  HEAD as a "new branch" and the failure only surfaces later in
  `gh pr create`. `CONTRIBUTING.md` gains the matching cultural rule
  (don't pipe `git commit` through anything that swallows its exit code).
  `pre-commit install` covers both stages now via
  `default_install_hook_types`.

### Documentation

- ADR-029 — *defer Tool 1 schema-field refinement to v0.6*. Phase 1's
  schema audit found one missing field (`FailedGuard.source_location`) and
  one nice-to-have (`FailedGuard.category`); both are deferred to the v0.6
  bump rather than churning a `v0.5.1` mid-MVP. Tool 1 derives source
  attribution at analyze time; the limitations are called out in the
  forthcoming Tool 1 docs page.

### Tests

- `tests/fixtures/recompile/` — corpus for the Tool 1 test suite. Five real
  PyTorch session captures (`simple_batch_size.log`, `mixed_guards.log`,
  `large_storm.log`, `tlparse_output/`, `dynamo_explain_output.json`),
  each paired with an `*.expected.json` oracle of the analyzer outcome it
  is meant to drive. `_generate.py` is the source of truth and
  regenerates the fixtures from `torch == 2.12.0+cpu` + `tlparse == 0.4.3`.
  Absolute paths, ISO timestamps, and PIDs are scrubbed so the fixtures
  diff byte-clean across machines.
- `crates/cls-cli/tests/collect_cli.rs` — six subprocess integration tests
  pinning the `cl collect` / `cl recompile-summary` exit-code contract
  (3 for `CLS-E0001`, 6 for `CLS-E0004`, 13 for `CLS-E0011`).
- `crates/cls-analyzer/tests/recompile_basic.rs` — two unit tests pinning
  the `recompile::analyze` surface (empty session → default findings;
  non-empty session → `NotYetImplemented`).
- `python/tests/collectors/test_recompile.py` — six tests pinning the
  `RecompileCollector` core: construction defaults, the additive
  `add_records` pipeline, `finalize()` output validated against both the
  pydantic binding and the `schema/v0.5.0.json` oracle (via the new
  `jsonschema` dev dependency), and the unknown-mode `ValueError`.

### Added

- `CLS-E0011 NotYetImplemented` error variant — a typed signal that a
  surface exists in the CLI / public API but its implementation lands
  later. Distinguishable from `AnalyzerInternalError`, which signals a
  bug in *existing* logic. Carries `surface` (the invoked surface) and
  `tracking` (the upcoming PR / issue) so the rendered diagnostic tells
  the user where to look. Exit code 13.
- `ClsError::exit_code()` — inherent variant-to-exit-code mapping
  (`3..13` for `CLS-E0001..CLS-E0011`). `0` is reserved for success; `2`
  is reserved for clap's argument-parse exit. A registry test pins the
  mapping so adding a variant without an exit code fails CI.
- `cl collect` is extended with `--from-logs <PATH>`, `--from-tlparse
  <DIR>`, `--from-dynamo-explain`, `--output <PATH>`, `--iterations <N>`
  (Phase 2 placeholder), and `--redaction <LEVEL>`. The mode flags are
  mutually exclusive (clap `group = "mode"`); supplying none gives
  `CLS-E0004` (exit 6); a nonexistent file gives `CLS-E0001` (exit 3);
  a successfully parsed but not-yet-implemented mode gives `CLS-E0011`
  (exit 13).
- `cl recompile-summary <SESSION>` subcommand with `--format
  markdown|json|text`. Same exit-code shape as `cl collect`.
- `cls-analyzer::recompile` module — `RecompileAnalyzer`,
  `RecompileFindings`, `GuardCategory`, `Suggestion` types plus
  `analyze(&Session) -> Result<RecompileFindings, ClsError>`. Empty
  sessions return default findings; non-empty sessions return
  `NotYetImplemented` until the upcoming Phase 1 PRs land the
  clustering and suggestion logic.

## [0.5.0-alpha.0] — 2026-05-31

First tagged release. Phase 0 (architectural scaffold) closeout — there is no
user-visible toolkit yet, only the foundation that the v0.5.0 MVP will land on
top of (Tool 1 recompile aggregator, Tool 2a compile diff, Hero `cl.session()`).

### Added

- **Schema v0.5.0** — JSON Schema (`schema/v0.5.0.json`), Rust serde bindings
  (`cls-schema` crate), Python pydantic models (`compile_lens._schema`). The
  `.cls.json` artifact is the sole cross-language contract.
- **Cross-language round-trip test** — Python writes → Rust reads → identical
  (and reverse); per-test byte-equality plus a determinism job that runs the
  whole suite twice at job level.
- **`cls-cli` (`cl`) binary** — `--version`, `cl collect <path>` (skeleton that
  exercises the error pipeline), `cl migrate <input> --output <out> | --dry-run`.
- **`cls-errors` crate** — 10 variants with stable codes (`CLS-E0001`..`E0010`)
  using `thiserror` + `miette` 4-part rendering (code + message + cause chain +
  help). ADR-022 documents the choice via a weighted matrix.
- **`cls-schema-migrate` crate** — detect-and-refuse migration skeleton. Pre-V1
  there is no migration ladder; matching the current schema is a byte-copy,
  everything else is refused (CLS-E0003) and the user re-collects.
- **Forward-compatible unknown-field capture** (ADR-027) — `ClsArtifact` and
  `Session` preserve unknown keys through a read/write cycle, every other type
  drops them. Paired `test_unknown_field_handled` on both language sides.
- **Tracing** — `tracing_subscriber` in `cls-cli` and `structlog` in the Python
  front-end, both configured by `CLS_LOG` (verbosity) / `CLS_LOG_FORMAT=json`
  (formatter); the Python side additionally accepts `CLS_DEBUG=1` as a force-
  debug shortcut.
- **CI** — per-PR matrix on Linux + macOS: `fmt + clippy`, `ruff + mypy`,
  `pytest` × Python 3.11/3.12 + Rust unit tests, determinism (round-trip ×2),
  migration (byte-copy + dry-run), tracing env-var contract. Weekly
  `cargo-audit` + `pip-audit` + dependabot for cargo / pip / github-actions.
- **Pre-commit hooks** mirroring the CI gates: trailing-whitespace,
  end-of-file-fixer, YAML/JSON/TOML well-formedness, merge-conflict markers,
  500 KB file-size ceiling, `ruff --fix` + `ruff-format`, `cargo fmt` and
  `cargo clippy -D warnings`.
- **GitHub tooling** — PR template carrying the 8-item engineering checklist
  (D7 algorithm / D8 error UX / D9 observability / D10 migration / D11 security
  / CI / API stability / perf); bug + feature issue templates; security and
  Discussions contact links.
- **Security docs** — `SECURITY.md`, threat model, redaction policy.

### Deferred to later phases (explicit)

- The torch matrix axis on CI (no `torch.compile` collector touches torch yet).
- A GPU smoke CI job (no GPU runner; smoke tests live locally for now).
- The nightly full matrix (no surface that would benefit from it pre-Phase 1).
- A typed schema migration ladder (kept as detect-and-refuse until V1, per the
  pre-V1 D10 exception in the design doc).

[Unreleased]: https://github.com/notAnIssue/compile-lens/compare/v0.5.0-alpha.0...HEAD
[0.5.0-alpha.0]: https://github.com/notAnIssue/compile-lens/releases/tag/v0.5.0-alpha.0
