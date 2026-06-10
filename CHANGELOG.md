# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `cl diff` is wired — Tool 2a (compile-diff) now runs end to end. `cl diff --base before.cls.json
  --head after.cls.json --format markdown|json|text` reads two collected sessions, diffs their
  compiled graphs with the WL-signature matcher, and renders the result: the nodes a change added /
  removed / modified, the matched pairs with a per-match confidence, and two quality numbers (match
  coverage and anchor uniqueness). A non-commutative operand swap shows up as `modified`, a
  commutative one stays silent, and a pure rename matches cleanly. (The subcommand previously parsed
  its arguments but returned `NotYetImplemented` while the algorithm was built up.)
- `CompileArtifactCollector` (Python) — the capture side of Tool 2a (compile-diff). It hooks a
  `torch.compile` run and serializes the **aten-normalized** FX graph into the inline
  `compiled_graphs[].nodes[]` contract (ADR-024): each node's `id`, `op_type`, ordered `inputs`,
  and scalar `attrs`. Routing through `aot_autograd` yields canonical aten ops (`aten.sub.Tensor`)
  rather than dynamo-level Python builtins, and operand order is preserved (load-bearing for the
  diff). Single-graph capture, `default-strict` redaction at write time; the diff consumer
  lands in a later PR. (`torch` stays an optional dependency, imported lazily.)
- Multi-iteration capture in `CompileArtifactCollector` — `capture(..., iterations=N)` runs the
  compiled function N times and records a per-run `iterations[]` entry: `cache_hit` /
  `recompilation_triggered` (detected by whether the run drove a fresh compile), an
  `output_signature` (sha256 over the run's output tensors — stable across iterations means
  identical output), and, for an `nn.Module`, an `internal_state_snapshot` (buffers hash + the
  scalar attributes that drifted since the previous iteration). This is the runtime-behaviour
  data foundation for cache-stability analysis. (Guard-evaluation capture is left for a later
  change.)
- FX-graph node representation for diffing (ADR-024). `compiled_graphs[]` gains an optional
  `nodes[]` array — each node carrying `id`, `op_type`, ordered `inputs` (the upstream node
  ids it consumes; order is load-bearing so `sub(a, b)` ≠ `sub(b, a)`), and free-form `attrs`.
  This is the node-level structure the WL-signature diff (Tool 2a) consumes. It is **inlined**
  into the artifact rather than pointing at a side file so a single `.cls.json` stays
  self-contained, and it is optional and forward-compatible: Tool 1 artifacts written before
  this field simply omit it. Mirrored across the Rust (`cls-schema`) and Python
  (`compile_lens._schema`) bindings and the JSON Schema, with cross-language round-trip parity.
- `cl diff` subcommand skeleton (Tool 2a — compile-diff). Parses `--base` / `--head`
  (both required) and `--format`, validates both artifact paths are reachable (a missing
  one reports `CLS-E0001`), and exits with the typed `NotYetImplemented` (`CLS-E0011`) until
  the WL-signature diff algorithm lands in the upcoming Phase 2 PRs — mirroring how the
  Tool 1 subcommands were stubbed before their analyzers existed. (The `--include-cache-stability`
  flag is intentionally **not** added yet — its logic is Tool 2b in a later phase.)

## [0.5.0-alpha.1] — 2026-06-08

Tool 1 (`recompile-summary`) complete: collect → analyze → render, end-to-end from the CLI,
with a regression-diff mode and a perf-characterized analyzer. (Still pre-MVP — v0.5.0 also
needs Tool 2a + the hero form.)

### Added

- `cl collect` is wired — the `cl` console script's `collect` subcommand now drives
  `RecompileCollector` to write a `.cls.json` instead of printing "not implemented yet".
  `--from-logs <path>` (Mode A) and `--from-tlparse <dir>` (Mode B) run from the CLI;
  `--from-dynamo-explain` (Mode C) is programmatic-only and prints how to use the Python API.
  `--output` is required and `--redaction` defaults to `default-strict` (the invocation is
  recorded as `session.command`, scrubbed at write time). This closes the loop: `cl collect`
  produces the artifact that `cl recompile-summary` analyzes.
- Recompile-summary output rendering + wired CLI — `cl recompile-summary <session.cls.json>`
  now runs Tool 1 and prints a report instead of returning `NotYetImplemented`. Three
  `--format` shapes via `cls_analyzer::recompile_render`: `markdown` (default, the
  human-readable summary from the design doc — total + per-axis guard clusters with their
  value transitions + ranked suggestions), `json` (the findings struct serialized verbatim,
  the machine-readable contract), and `text` (plain layout for non-Markdown terminals).
  Adding `--baseline <earlier.cls.json>` switches to the regression diff (added / grown /
  removed clusters). Both modes share one loader, `cls_schema_migrate::load_artifact` — the
  read + schema-version-gate + parse seam reused by the `--baseline` load and the future
  compile-diff tool, so artifact loading has one home. Rendering lives only in
  `recompile_render` (the analyzer returns plain data; presentation is a separate concern).
- Session-to-session recompile diff (Tool 1 regression mode) — `cls_analyzer::recompile_diff::diff_recompiles(base, head)`
  answers the comparison-axis question the single-snapshot analyzer can't: *what did
  this commit/PR change about recompile behaviour vs a baseline?* It clusters both
  sessions (reusing the clustering) and classifies into `added` (new recompile axes
  this change introduced), `grown` (same axis, more recompiles — with the base→head
  delta), and `removed` (fixed). Suggestions are **regression-anchored** ("this change
  introduced N recompiles on `x[0]` → mark_dynamic(x, 0)"), which the before/after diff
  makes defensible. Takes two parsed `ClsArtifact`s; baseline *loading* is deliberately
  left to the caller so it can be shared with the upcoming compile-diff tool (one
  artifact-pair loading path, not two).
- Top suggestions (Tool 1 prescribe step) — `cls_analyzer::recompile_suggest` turns each
  actionable guard cluster into one **axis-precise** `Suggestion`: a `size` cluster on
  `x[0]` yields `torch._dynamo.mark_dynamic(x, 0)` (not a vague "consider marking something
  dynamic"), `dtype` → "pin the dtype upstream", `stride` → "insert `x.contiguous()`".
  Suggestions are **N2**: rejectable hints with an `evidence` trace back to the cluster
  (category, axis, count, value transitions), never auto-applied patches; the `other`
  (unrecognized) category yields no suggestion rather than a noisy one. They inherit the
  cluster ranking (most-frequent recompile first). `analyze` now fills `top_suggestions`;
  `Suggestion` grows an `evidence` field.
- Guard clustering (Tool 1 analyzer core) — `cls_analyzer::recompile::analyze` now
  clusters a session's recompiles instead of returning `NotYetImplemented`. Following
  the describe→attribute north star (ADR-032), it parses each failed guard's structured
  text (`tensor 'x' size mismatch at index 0`) into a category (`size`/`dtype`/`stride`,
  else `other`), a canonical template, and a **dynamic axis** (`x[0]`), then groups by
  `(category, axis)` so each actionable axis is its own cluster (`GuardCategory` with
  count + distinct `previous→new` value transitions). Output is deterministic
  (count-desc, then template, then axis). `analyze` now takes `&ClsArtifact` (the
  recompiles live on the parent artifact, not the nested `Session`). Hand-parses the
  guard text rather than adding a regex dependency (ADR-015 right-sizing); the clustering
  algorithm choice is recorded inline in the module (no standalone ADR: the structured
  guard text made the anticipated token/AST/edit-distance clustering choice moot).
- Collect-time redaction (D11) — `compile_lens.security.redactor`: `scrub_command`
  (argv token scrub), `normalize_path` (collapse user/install paths to `{repo}` /
  `{torch_install}` placeholders and **refuse** a credential-dir path by raising
  `SensitivePathError` / `CLS-E0008`), `hash_host` (FQDN → stable `dh-<hash>`),
  `filter_env_vars` (whitelist torch.compile vars, denylist secrets). `RecompileCollector`
  now applies these in `finalize` under a strict (`default-strict` / `public-safe`)
  policy — the command and host are scrubbed and compiled-graph paths normalized at
  write time, so a `default-strict` artifact never holds a raw secret; `internal` /
  `confidential` keep raw fields. The host salt comes from `~/.compile-lens/install-id`
  (overridable via `CLS_INSTALL_ID`). `RecompileCollector` gains a `host` parameter.
- Mode B collector — `compile_lens.collectors.tlparse_adapter.parse_tlparse_dir`
  reads a `tlparse` output directory (its `compile_directory.json` compile-id
  index + `raw.jsonl` events) into one `CompiledGraph` per compile (with the
  dynamo/inductor artifact paths + the symbolic `guard_added_fast` guards) and
  one `Recompilation` per compile with counter ≥ 1 (attribution + backend
  compile time). It wraps tlparse rather than re-parsing the trace (P8); a
  malformed `raw.jsonl` line is skipped, not raised. `RecompileCollector.from_tlparse`
  is wired to it.
- Mode C collector — `compile_lens.collectors._dynamo_explain_adapter.parse_dynamo_explain`
  adapts a `torch._dynamo.explain` result (the live object or its serialized
  dict) into `GraphBreak`s (one per break reason) + `CompiledGraph`s (one per
  graph). It is a single-run structural view, so it never contributes
  `recompilations`. `RecompileCollector.from_dynamo_explain` is wired to it.
- `RecompileCollector.add_records` / `finalize` now also carry `graph_breaks`
  (needed by Mode C); an artifact with no graph breaks stays byte-identical.
- Mode A collector — `compile_lens.collectors._logs_parser.parse_recompiles_log`
  turns a `TORCH_LOGS=recompiles` text dump into `Recompilation` records (one
  per recompile block: compile id, function, primary failed guard with
  expression + previous/new value, recompile ordinal). It keys off the
  `[__recompiles]` marker and ignores the version-specific log prefix, and
  skips malformed lines with a warning rather than raising. `RecompileCollector.from_logs`
  is wired to it, so `cl collect --from-logs` / `collect("logs", path)` now
  produce a populated `.cls.json`.
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

- Recompile clustering value-transition dedup is now O(1) per event (was O(n²)). Distinct
  `previous → new` transitions per cluster are collected into an `IndexSet` instead of a `Vec`
  with a linear `contains` scan — a recompile storm can cycle through thousands of distinct
  values, and the benchmark added in this release showed the old dedup made a 10× larger input
  ~86× slower. Output is unchanged (same distinct transitions, same insertion order, so findings
  stay byte-identical); 10k-event analysis dropped from ~42ms to ~3.5ms and per-event throughput
  is now ~constant across input sizes.
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

### Fixed

- Mode A parser: non-shape guards are now parsed cleanly. torch appends a
  ` # <source> # <file>:<line>` comment after value/boolean guards
  (`n == 1`, `G['flag'] == True`); the parser now strips it from
  `failed_guard.expression` (it was leaking into the expression and breaking
  clustering of the same guard across call sites). The `- User stack trace:`
  continuation block that follows a failure is skipped silently instead of
  logging an "unrecognized line" warning per frame. Documented that
  `failed_guard.previous_value` / `new_value` are genuinely optional — only
  tensor shape/dtype/stride mismatches carry a before/after value; boolean
  and value guards carry none. (The shape-only fixture corpus had hidden
  this; see the new `nonshape_guards` fixture.)

### Documentation

- `docs/03_tools/recompile_summary.md` — Tool 1 user guide (Part A theory & reference:
  describe→attribute→prescribe, the three collection modes, guard categories, findings shape;
  Part B examples: Markdown/JSON/`--baseline` diff CLI, Python collector API, a GitHub Action
  regression-guard template — all runnable against the committed `schema/examples/`; Part C FAQ,
  limitations, and the "PyTorch nightly changed the log format" failure case). README gains a
  working `recompile-summary` demo with real output.
- ADR-029 — *defer Tool 1 schema-field refinement to v0.6*. Phase 1's
  schema audit found one missing field (`FailedGuard.source_location`) and
  one nice-to-have (`FailedGuard.category`); both are deferred to the v0.6
  bump rather than churning a `v0.5.1` mid-MVP. Tool 1 derives source
  attribution at analyze time; the limitations are called out in the
  forthcoming Tool 1 docs page.

### Tests

- Tool 1 end-to-end integration tests + performance benchmark. `tests/integration/test_tool1_e2e.py`
  drives the full cross-language pipeline — fixture log / tlparse dir → Python collector →
  `.cls.json` → `cl recompile-summary --format json` → assert against each fixture's committed
  semantic oracle (recompile count, surfaced cluster categories, raw guard text, suggestion
  keywords) — across 5 scenarios (4 Mode-A log fixtures + 1 Mode-B tlparse dir).
  `crates/cls-analyzer/benches/recompile_summary.rs` is a criterion benchmark over 100 / 1k / 10k
  recompiles (throughput-reported); `crates/cls-analyzer/tests/recompile_perf.rs` is the hard
  deterministic gate (1000-event analyze < 1s, 10k scales without going super-linear) so CI
  catches a perf regression without running the full statistical bench.
- `tests/fixtures/recompile/` — corpus for the Tool 1 test suite. Five real
  PyTorch session captures (`simple_batch_size.log`, `mixed_guards.log`,
  `large_storm.log`, `tlparse_output/`, `dynamo_explain_output.json`),
  each paired with an `*.expected.json` oracle of the analyzer outcome it
  is meant to drive. `_generate.py` is the source of truth and
  regenerates the fixtures from `torch == 2.12.0+cpu` + `tlparse == 0.4.3`.
  Absolute paths, ISO timestamps, and PIDs are scrubbed so the fixtures
  diff byte-clean across machines.
- `tests/fixtures/recompile/nonshape_guards.log` (+ oracle) — a non-shape
  guard recompile (`n == 1`, a python-int specialization) with torch's trailing
  source comment and a `- User stack trace:` block; the case the shape-only
  corpus was missing. `_generate.py` grows a `nonshape_guards` workload.
  `test_logs_parser.py` gains 4 tests (comment stripping, no-value guards,
  `requires_grad` expected-without-actual, stack-trace skipping).
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
- `python/tests/collectors/test_logs_parser.py` — 13 tests for the Mode A
  parser: the three committed fixtures (simple / mixed / large_storm), focused
  size/dtype/stride extraction, base-compile primary-guard selection, prefix
  robustness across torch-version log formats, malformed-line skipping, and
  the `from_logs` end-to-end into a valid `.cls.json`.
- `python/tests/collectors/test_tlparse_adapter.py` — 8 tests for Mode B
  against the committed `tlparse_output/` fixture (3 recompiles / 4 graphs):
  recompile detection, compiled-graph artifact paths, guard mapping, wall-clock
  from metrics, missing-file + malformed-line + partial-capture tolerance, and
  `from_tlparse` end-to-end.
- `python/tests/collectors/test_dynamo_explain.py` — 5 tests for Mode C: the
  no-break fixture, a with-breaks dict, the live-object shape, the
  no-recompilation guarantee, and a `graph_breaks` round-trip validated against
  the `schema/v0.5.0.json` oracle.
- `python/tests/security/test_redactor.py` — 14 tests for collect-time redaction:
  path normalization (home / conda) + credential-path refusal (`~/.ssh`, `~/.aws`,
  `~/.gnupg`), argv token scrub (hf / openai / bearer / aws), FQDN hashing,
  env whitelist/denylist, and the collector wiring (strict scrubs command + host,
  `internal` keeps raw, sensitive compiled-graph path refused at finalize).
- `crates/cls-analyzer/tests/recompile_cluster.rs` — 8 tests for guard clustering:
  single / multi-category, canonical-template + axis extraction, axis-level separation
  (`x[0]` vs `x[1]`), deduped value transitions, `other`-category literal canonicalization,
  malformed-guard skipping (still counted), and determinism. `recompile_basic.rs` updated
  to the `&ClsArtifact` signature.
- `crates/cls-analyzer/tests/recompile_suggest.rs` — 7 tests for suggestions: axis-precise
  `mark_dynamic(x, 0)`, dtype + contiguous advice, `other` yields none, ranking by recompile
  count, the `evidence` trace carries count + value transitions, and the empty case.
- `crates/cls-analyzer/tests/recompile_diff.rs` — 7 tests for the regression-mode diff:
  added / grown (with base→head delta) / removed classification, identical-sessions empty
  diff, regression-anchored suggestions for added + grown (the "this change introduced N"
  headline), and `other`-category regression yielding no suggestion.

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
