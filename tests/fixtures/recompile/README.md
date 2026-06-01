# Recompile fixture corpus

Input/output pairs the Tool 1 (*recompile aggregator*) test suite consumes.
Each fixture is a real PyTorch session capture — there are no fabricated log
formats here — paired with an `*.expected.json` oracle of what the Tool 1
analyzer is expected to derive from it.

## Inventory

| Fixture | Bytes | What it exercises |
|---|---|---|
| `simple_batch_size.log` | 65 KB | A single guard cluster — one dimension of one tensor — captured under `TORCH_LOGS=+recompiles`. Smallest end-to-end Mode A case. |
| `mixed_guards.log` | 130 KB | Three guard categories — size, dtype, stride — in one session. The canonical Mode A clustering input. |
| `large_storm.log` | 65 KB | ~100 distinct shapes producing the full per-event content the perf benchmark needs as a unit. The benchmark inflates to ≥ 1000 events at test time by iterating, so the committed fixture stays under the 500 KB pre-commit ceiling. |
| `tlparse_output/raw.jsonl` | 182 KB | Line-delimited structured events produced by `tlparse` against a `TORCH_TRACE` capture of the same workload as `mixed_guards`. Mode B's primary parse target. |
| `tlparse_output/compile_directory.json` | 11 KB | The structured directory index `tlparse` writes alongside `raw.jsonl`. Mode B uses it for compile-id navigation. |
| `dynamo_explain_output.json` | < 1 KB | The serialized fields of a `torch._dynamo.explain` result for the mixed-guards model. Mode C's input. |

`*.expected.json` files sit next to each fixture and describe the analyzer
outcome the fixture is meant to drive. They are oracles, not schema-validated
analyzer output — the analyzer ships in later PRs, and the oracles document
what *correctness* looks like before the implementation is allowed to define
it.

## How they were produced

`_generate.py` is the source of truth. To regenerate:

```bash
# from repo root, with torch + tlparse installed in the venv
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tlparse
python tests/fixtures/recompile/_generate.py
```

The script:

1. Runs three workloads under `TORCH_LOGS=+recompiles`, captures the
   structured-logger output, scrubs absolute paths and timestamps, and
   writes `simple_batch_size.log`, `mixed_guards.log`, `large_storm.log`.
2. Runs the `mixed_guards` workload a fourth time under `TORCH_TRACE=<tmp>`,
   feeds the resulting trace file to the `tlparse` CLI, and keeps only the
   files Mode B will parse (`raw.jsonl` + `compile_directory.json` — the
   per-compile HTML / FX-graph sub-directories add ~1 MB of artifact Mode B
   does not consume).
3. Calls `torch._dynamo.explain` on the mixed-guards model and writes the
   serialized fields into `dynamo_explain_output.json`.

When the PyTorch log format drifts (it does, across minor versions),
re-running `_generate.py` shows the diff against committed fixtures, and the
expected oracle JSON files are reviewed for whether the semantic outcome
still matches.

## Determinism

Absolute paths are replaced with `<REPO>` and `<HOME>`. ISO timestamps are
collapsed to `<TS>`. PIDs are replaced with `<PID>`. File mtimes inside
`tlparse_output/` are pinned to the epoch. The fixtures should diff
byte-clean across machines as long as torch and tlparse versions match.

The torch + tlparse versions the fixtures were generated against:

- `torch == 2.12.0+cpu`
- `tlparse == 0.4.3`

Bumping either is a deliberate act: do it in a PR that also re-runs
`_generate.py` and reviews the diff.
