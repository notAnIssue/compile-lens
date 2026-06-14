# Tool 5 — `kernel-roofline`

Triage a Triton kernel against the roofline before autotuning it, and prune the autotune search with
a **measured guarantee**. For each kernel it answers two questions: *is this kernel already near the
hardware ceiling* (so autotuning can't help), and *which of an autotune grid's configs are worth
actually measuring*.

`kernel-roofline` is a **roofline-pruned autotune filter, not a performance predictor**. Absolute
accuracy is left to `proton` / `ncu`; the cost model only ranks configs cheaply so an autotuner can
skip the ones that can't win. Its differentiator is the *measured* guarantee: it calibrates its
predicted ranking against a small sample of real measurements and only prunes when that calibration
holds — otherwise it falls back to measuring everything. It never silently trusts a bad prediction.

## Part A — Theory & reference

The cost model has three layers. Layers 1 and 2 are pure arithmetic (no GPU); Layer 3 needs a small
sample of real measurements.

### Layer 1 — theoretical lower bound (Williams roofline)

From a kernel's FLOPs `F` and moved bytes `B`, against a GPU's peak compute `C` (FLOP/s) and peak
bandwidth `M` (bytes/s):

- **arithmetic intensity** `AI = F / B` (FLOP per byte);
- **ridge point** `AI* = C / M` — the intensity where the compute and bandwidth ceilings meet. Below
  it a kernel is *memory-bound*; at or above it, *compute-bound*;
- **theoretical lower-bound time** `max(F / C, B / M)` — the larger of the compute-limited and
  bandwidth-limited times. No kernel can beat this on this GPU.

This layer is a **reference** — "here is the floor" — and is deliberately *not* used for pruning
decisions. Reporting a sound lower bound while also pruning on a predictor that can exceed it would
be self-contradictory, so the layers are kept separate (ADR-018).

### Layer 2 — empirical predictor (four corrections)

Real kernels pay more than the ideal floor. Layer 2 inflates the lower bound into a predictor used
for **ranking**:

```
empirical_us = lower_bound_us × (1 + block_size_penalty) × (1 + occupancy_penalty)
             + launch_overhead_us + register_pressure_penalty_us
```

1. **block size** — a block too small to saturate the memory pipeline pays a penalty that ramps up
   below a saturation threshold.
2. **occupancy** — register-limited occupancy (the NVIDIA occupancy method); low occupancy can't
   hide latency, so the penalty is the inverse-occupancy shortfall, capped.
3. **launch overhead** — a fixed additive per-launch cost; it dominates for tiny kernels.
4. **register pressure** — spilling adds local-memory traffic. **Softened (ADR-018):** a spill goes
   to *local memory*, which is usually L1/L2-cached and reaches HBM only on a miss, so it is a
   *modest* amplifier on register-bound kernels — not the "doubles bandwidth, biggest factor on
   H100" claim an earlier draft made. The penalty is capped well below 2×.

The contract here is a **monotonic ranking**, not µs accuracy. The coefficients are documented
defaults; the model earns trust through Layer 3's correlation against real measurements, not through
these constants being exact.

### Layer 3 — calibrated pruning (rank-correlation gated)

Autotuning is a *ranking* task — keep the top-K fastest — so the model is validated by **rank**
correlation (Spearman), not Pearson (ADR-018). A high Pearson correlation can coexist with a wrong
top-K order (the relationship is linear-ish overall but the fastest few are mis-ordered); rank
correlation measures the agreement that actually matters for pruning. Three tiers on the Spearman
correlation between predicted and measured runtimes:

| Rank correlation | Tier | Action |
|---|---|---|
| ≥ 0.8 | `aggressive` | trust the ranking — measure only the predicted top-K |
| 0.5 – 0.8 | `moderate` | partial trust — measure the top half |
| < 0.5 (or undefined) | `disabled_fallback_full_sweep` | the model violates its assumptions here — measure everything |

The third tier is the safety net: when the prediction can't be trusted, the tool measures all configs
rather than risk pruning away the genuinely fastest one. This is what makes the pruning *safe* — the
省时 is real because it is gated on a real correlation.

### The cost model is single-sourced in Rust

The three layers live once, in the `cls-roofline` crate. The `cl kernel-roofline` CLI and the Python
`AutotuneHarness` both reach it the same way: by writing a `.cls.json` and reading the analysis back
over a subprocess (ADR-006 — the cross-language boundary is a JSON file, never an in-process
binding). There is no second copy of the model in Python.

## Part B — Examples

### CLI

```bash
# List the kernels captured in a session.
cl kernel-roofline session.cls.json --list

# Roofline triage for every kernel against an H100, as a human-readable table.
cl kernel-roofline session.cls.json --gpu H100-SXM

# One kernel, machine-readable (what the autotune harness consumes).
cl kernel-roofline session.cls.json --gpu A100-SXM-80GB --kernel sgemm --format json --top-k 2
```

When a session's kernels carry measured runtimes, the report adds a grid-level calibration and
pruning decision; with features only, it reports the per-kernel predictions alone.

#### Options

- `--gpu <NAME>` — the GPU spec to compute the roofline against (registry name, case-insensitive;
  default `A100-SXM-80GB`). The registry ships verified **data-center** specs (A100-SXM-80GB /
  H100-SXM / B200); consumer GPUs are intentionally omitted until their dense-tensor throughput is
  verified, since a guessed peak would bias every prediction.
- `--kernel <SUBSTR>` — restrict the rendered report to kernels whose id contains the substring (real
  kernel names are mangled, so a substring like `sgemm` is enough).
- `--list` — print the session's kernel ids and names, then exit without analyzing.
- `--top-k <K>` — the number of configs to keep under aggressive pruning (default 5).
- `--format markdown|json` — `markdown` (default) is the human table; `json` is the machine shape the
  harness reads.

#### Worked example — pruning a well-ranked grid

A four-config grid where the predictor ranks the configs the way the measured runtimes do:

```
## Kernel Roofline (A100-SXM-80GB)

| kernel        | bound  | lower µs | predicted µs | measured µs | corrections                 |
|---------------|--------|----------|--------------|-------------|-----------------------------|
| cfg_bs256_r32 | memory | 617.9    | 622.9        | 650.0       | launch_overhead             |
| cfg_bs64_r32  | memory | 617.9    | 777.4        | 820.0       | block_size, launch_overhead |
| cfg_bs32_r32  | memory | 617.9    | 854.7        | 900.0       | block_size, launch_overhead |
| cfg_bs256_r64 | memory | 617.9    | 1240.9       | 1300.0      | occupancy, launch_overhead  |

**Pruning:** mode `aggressive` — measure 2 of 4 configs (rank correlation 1.000).
```

The predicted order matches the measured order, so the rank correlation is 1.0, the tier is
`aggressive`, and an autotuner would measure only the predicted top 2 of 4.

### Python autotune harness

The `AutotuneHarness` drives the same analysis to prune a live autotune sweep — predict every config,
measure a small calibration sample plus the predicted top-K, and report the fastest:

```python
from compile_lens.kernels import AutotuneHarness

harness = AutotuneHarness(
    config_grid={"BLOCK_SIZE": [64, 128, 256, 512], "num_warps": [4, 8]},
    features_for=my_features_fn,   # config -> kernel features (flops/bytes/block_size/num_regs)
    run_for=my_run_fn,             # config -> a zero-arg callable that runs the kernel once
    gpu="H100-SXM",
    keep_top_k=4,
)
report = harness.sweep()           # predict (Rust) -> calibrate -> measure only what the tier allows
print(report.best_config, report.pruning_ratio)
report.to_csv("autotune.csv")
report.plot_predicted_vs_measured("calibration.png")   # matplotlib optional
```

The winner is always the lowest **measured** config (the prediction is the filter, the measurement is
the judge). Measurement uses `proton` when a trace is available and falls back to self-timed CUDA
events otherwise; the source is recorded as `measurements.source` (`proton` / `self_timed`).

## Part C — Limitations

A roofline cost model is a *filter*, and it is honest about the kernels it cannot rank.

- **It cannot rank a kernel it has no signal for — and correctly says so.** The model's ranking
  signal comes from block size, occupancy, and register spill. A purely memory-bound kernel whose
  configs share the same FLOPs and bytes (e.g. an elementwise kernel across block sizes ≥ the
  saturation threshold) has the *same* lower bound and *no* block-size or occupancy penalty, so the
  model predicts every such config identically. The rank correlation is then undefined, and Layer 3
  falls back to a full sweep — measuring all configs rather than pruning on a ranking it does not
  have. This is the design working as intended: the pruning is only as aggressive as the correlation
  earns, and a kernel the model can't rank costs nothing in safety, only the time it would have taken
  to measure anyway.
- **Prediction is GPU-free; measurement is not.** Layers 1 and 2 are arithmetic and run anywhere
  (CI computes them from recorded features). Layer 3's calibration and the harness's sweep need a GPU
  to measure on; without one, the analysis still produces predictions but cannot calibrate. The
  algorithmic correctness of the pruning gate is guarded deterministically in CI; the question of how
  well the model ranks any *particular* real kernel is empirical and measured locally.
- **Rank correlation, not Pearson** (ADR-018) — see Layer 3. Pruning is a ranking decision, so it is
  gated on a ranking statistic.
- **Modeled, not learned.** The four corrections assume coalesced access and ignore cache thrashing
  beyond the register-pressure term. Tensor-Core ridge points (FP16/BF16 vs FP32), software-pipeline
  stages (Hopper `wgmma`), and a hierarchical L1/L2/HBM roofline are deliberately out of scope for
  now (post-v1.0 candidates). The model predicts mean runtime, not tail latency, and does not replace
  a learned cost model — it is a cheap, transparent filter.

## Related

- `AutotuneHarness` (Python) — the calibrated-pruning sweep built on this analysis.
- Tool 6 (CODA fusion detector) builds on the same `RooflineCostModel` to estimate fused-vs-baseline
  HBM traffic.
- `proton` (Triton's profiler) is the primary measurement source; self-timed CUDA events are the
  fallback.
- ADR-018 (three-layer separation, rank-correlation gating, spill softening).
