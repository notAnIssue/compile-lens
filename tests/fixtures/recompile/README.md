# Recompile fixture corpus

Input/output pairs the Tool 1 (*recompile aggregator*) test suite consumes.
Each fixture is a real PyTorch session capture — there are no fabricated log
formats here — paired with an `*.expected.json` oracle of what the Tool 1
analyzer is expected to derive from it.

## Inventory

| Fixture | Bytes | What it exercises |
|---|---|---|
| `simple_batch_size.log` | ~0.4 KB | A single size-guard recompile — one dimension of one tensor — as emitted by `TORCH_LOGS=recompiles`. Smallest end-to-end Mode A case. |
| `mixed_guards.log` | ~1.5 KB | Three guard categories — size, dtype, stride — in one session (3 recompiles). The canonical Mode A clustering input. |
| `large_storm.log` | ~178 KB | 49 size-guard recompiles (50 distinct 1-D shapes under `dynamic=False`). Each recompile lists its guard failure against every prior cache entry, so this is the realistic O(N²) listing the parser must chew through. The perf benchmark inflates the count at test time by iterating, so the committed fixture stays under the 500 KB pre-commit ceiling. |
| `tlparse_output/raw.jsonl` | 182 KB | Line-delimited structured events produced by `tlparse` against a `TORCH_TRACE` capture of the `mixed_guards` workload. Mode B's primary parse target. |
| `tlparse_output/compile_directory.json` | 11 KB | The structured directory index `tlparse` writes alongside `raw.jsonl`. Mode B uses it for compile-id navigation. |
| `dynamo_explain_output.json` | < 1 KB | The serialized fields of a `torch._dynamo.explain` result for the mixed-guards model. Mode C's input. |

Each `<scenario>.log` is exactly the `[__recompiles]` artifact stream that
`TORCH_LOGS=recompiles` writes to stderr — line shape
`V<TS> <PID> torch/_dynamo/guards.py:<lineno>] [<compile_id>] [__recompiles] <message>`
— which is the text a user sees when they enable the flag. The Mode A parser
keys off the `[__recompiles]` marker and parses the message after it, so the
scrubbed `<TS>`/`<PID>` placeholders in the prefix are irrelevant to parsing.

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

1. Runs each `.log` workload in a **subprocess** with `TORCH_LOGS=recompiles`
   set (the env var is read at torch-import time, so it must precede the
   import — an in-process `set_logs` after the parent already imported torch
   does not emit the same artifact). It keeps only the `[__recompiles]` lines
   from the child's stderr, scrubs the volatile prefix fields + the workload
   path, and writes `simple_batch_size.log`, `mixed_guards.log`,
   `large_storm.log`.
2. Runs a `mixed_guards`-shaped workload under `TORCH_TRACE=<tmp>`, feeds the
   resulting trace file to the `tlparse` CLI, and keeps only the files Mode B
   will parse (`raw.jsonl` + `compile_directory.json` — the per-compile HTML /
   FX-graph sub-directories add ~1 MB of artifact Mode B does not consume).
3. Calls `torch._dynamo.explain` on the mixed-guards model and writes the
   serialized fields into `dynamo_explain_output.json`.

> The Mode B (`tlparse_output/`) and Mode C (`dynamo_explain_output.json`)
> fixtures are independent capture paths with their own inline workload and
> are **not** touched when only the Mode A `.log` corpus is regenerated; they
> are refreshed when the Mode B / Mode C collectors land.

When the PyTorch log format drifts (it does, across minor versions),
re-running `_generate.py` shows the diff against committed fixtures, and the
expected oracle JSON files are reviewed for whether the semantic outcome
still matches.

## Determinism

In the `.log` fixtures the glog prefix's date+time is collapsed to `<TS>` and
the pid to `<PID>`; the workload's own file path is collapsed to `<REPO>`. The
`guards.py:<lineno>` source ref in the prefix is torch-version-specific but
stable for a given version. In `tlparse_output/` absolute paths are replaced
with `<REPO>` / `<HOME>` and file mtimes are pinned to the epoch. The fixtures
diff byte-clean across machines as long as torch and tlparse versions match.

The torch + tlparse versions the fixtures were generated against:

- `torch == 2.12.0+cpu`
- `tlparse == 0.4.3`

Bumping either is a deliberate act: do it in a PR that also re-runs
`_generate.py` and reviews the diff.
