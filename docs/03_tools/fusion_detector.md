# Tool 6 — `fusion-detect`

Find algebraic fusion opportunities a `torch.compile` graph leaves on the table — specifically the
`GEMM → residual add → RMSNorm → GEMM` chain ("Pattern A") — and estimate the HBM traffic a fused
kernel would save. Each opportunity carries its location, the GEMM shape, a baseline-vs-fused traffic
estimate, a speedup, a confidence, and a plain-language description of the fusion to apply.

`fusion-detect` is a **detector, not an optimizer**. It reads the serialized graph, never mutates it,
never runs a kernel, and needs no torch at analysis time — it points at opportunities, estimates
their value, and describes the fusion to apply; realizing the saving (writing the fused epilogue
kernel) is your job, out of scope for the detector. Treat the speedup as a *ranking signal*, not a
promise — see *Part C*.

## Part A — Theory & reference

### What it does

Pattern A is the cross-sublayer shape `GEMM₁ → (+ residual) → RMSNorm → GEMM₂`. RMSNorm scales each
row by a single per-row scalar `1/rms`. Because that scalar is constant across a row, it can be
pulled *through* the second GEMM and applied in its epilogue — so the full-tensor RMSNorm output
between the two GEMMs never has to be written to and re-read from HBM. Inductor's local scheduling
rules don't perform this cross-module algebraic fusion, so it is left on the table; this tool finds
where.

The detector is a Rust analyzer over the FX graph the collector already serialized into the
`.cls.json` (ADR-037) — there is no re-trace and no live Python graph. It recognizes the pattern from
two things in that graph: the **operator types** (`is_matmul` / `is_add` / RMSNorm) and the
**single-consumer topology** read off the edges. The single-consumer links are the safety condition:
a fusion is only sound to suggest if each intermediate tensor feeds *only* the next step in the
chain. A tensor that also escapes to some other consumer can't be folded away, so that opportunity is
(correctly) not reported.

### Native and decomposed RMSNorm

`nn.RMSNorm` rarely survives to a single `aten.rms_norm` node — `torch.compile` usually lowers it to
a **decomposed** subgraph (`pow → mean → [+ eps] → rsqrt → mul → [× weight]`). The matcher handles
both forms. An exact native `aten.rms_norm` match is reported at **high** confidence; a decomposed
subgraph match — the common case for real models — at **medium**.

### The cost model

The estimate is an analytical **HBM-traffic roofline**, never a measured runtime. The baseline path
(four separate kernels — GEMM₁, the residual add, RMSNorm, GEMM₂) streams the activation tensor
through HBM roughly eight times; the fused path streams it about three (the RMSNorm is never
materialized), plus a little fp32 traffic for the partial statistics. The reported speedup is the
ratio of those two byte counts under a memory-bound assumption. Bandwidth cancels in the ratio, so
the estimate needs no GPU.

This ratio is an **upper bound used to rank opportunities, not a number you will hit.** Real GEMMs are
often compute-bound, so the achieved speedup is lower — on the order of 1.05–1.15× on LLaMA-shaped
layers. The value of the estimate is *ordering*: which opportunity is worth your attention first.

### Applying a fusion (out of scope)

Each opportunity carries a plain-language `suggested_fusion` — for Pattern A, "fold per-row 1/rms into
the second GEMM epilogue". That is a *description of the transformation*, not a kernel reference: the
tool names no library, imports nothing, and calls nothing. Realizing the saving means writing (or
reusing) a fused epilogue kernel yourself and wiring it in. The detector's job ends at "here is a
foldable pattern worth ~Nx, and here is the fusion that folds it."

## Part B — Examples

The Python front-end captures a `torch.compile` run into a `.cls.json`; the Rust analyzer reads it.

```bash
cl fusion-detect session.cls.json
```

On a graph carrying a `GEMM-Residual-RMSNorm-GEMM` block (here a LLaMA-style MLP shape, M=8192,
K0=4096, N=11008, K1=4096, bf16):

```
## Fusion Opportunities Detected

### #1: GEMM-Residual-RMSNorm-GEMM
- Pattern: A (GEMM-Residual-RMSNorm-GEMM)
- Location: FX nodes g1, add, rms, g2
- Shape: M=8192, K0=4096, N=11008, K1=4096, dtype=bf16
- Baseline HBM traffic: 1757 MB
- CODA-fused HBM traffic: 861 MB
- Estimated speedup: ~2.04× (memory-bound upper bound — ranks opportunities, not a promise; real is lower, ~1.05–1.15× per the paper, GEMMs being compute-bound)
- Suggested fusion: fold per-row 1/rms into the second GEMM epilogue
- Confidence: high
```

### Options

- `--format markdown|json` — `markdown` (default) is the human report above; `json` is the
  machine-readable `{ "fusion_opportunities": [...] }`, the same shape the artifact carries, so it
  round-trips back into the schema.
- `--min-speedup <FLOAT>` — drop opportunities whose estimated speedup is below this floor (default
  `1.05`). An opportunity whose shapes weren't captured has *no* estimate and is always kept (it is a
  real finding that simply can't be ranked).
- `--top-k <N>` — keep at most this many opportunities after ranking by speedup (default `10`).

Opportunities are always ordered by estimated speedup, highest first.

### Exit code

`fusion-detect` is informational, not a gate: a successful read always exits `0`, however many
opportunities it found (contrast `compile-lint`, which exits `1` on a surviving `high`). A non-zero
exit means the tool itself failed — e.g. the session file could not be read.

## Part C — Limitations & failure cases

A detector that overstates its reach is worse than none. What this tool does *not* do:

- **One pattern.** Only `GEMM-Residual-RMSNorm-GEMM` (Pattern A) is matched today. Other epilogue
  fusions (pairwise activations, online-LSE scoring) are out of scope for now; extending the tool
  means adding a concrete named matcher, not generalizing to a rewrite engine.
- **The speedup is an estimate, not a measurement.** It is an analytical memory-bound *upper bound*
  for ranking. The realized speedup from an actual fused kernel is lower and must be measured
  separately — this tool never runs a kernel.
- **Concrete shapes only.** The cost needs tensor shapes. With dynamic shapes (`torch.compile`'s
  symbolic dims) there is no concrete byte count, so the opportunity is still reported but its
  shape/traffic/speedup are left blank rather than guessed.
- **Forward graphs only.** Training/backward subgraphs are never analyzed by design; a graph carrying
  a backward-marked op is skipped wholesale.
- **Suggest-only.** It reports, estimates, and describes the fusion; it does not apply it or call any
  kernel. Realizing the saving means writing the fused epilogue kernel yourself.
- **Conservative on escapes.** If an intermediate tensor in the chain also feeds something outside
  the pattern, the fusion isn't safe to fold, so the opportunity is not reported. This favors
  precision over recall — it may stay silent on a pattern that a human could fold with care.
- **It reports what's foldable, not what Inductor already folded.** The estimate is the traffic of
  the *unfused* baseline vs the *fully-fused* kernel. If Inductor already performed part of the
  fusion (e.g. kept an intermediate on-chip between the two GEMMs), the marginal saving from
  the epilogue fusion is smaller than the headline ratio. Measure before and after.

## Related

- `kernel-roofline` (Tool 5) — the analytical roofline approach this cost model shares; Tool 5 ranks
  autotune configs, Tool 6 ranks fusion opportunities.
- **CODA** (Guo et al., arXiv:2605.19269) — the GEMM-epilogue fusion technique this detector is
  modeled on (folding a row-wise reduction's reciprocal through the downstream GEMM's epilogue).
- ADR-037 — why Tool 6 is a Rust analyzer over the serialized FX graph rather than a live-Python pass.
- ADR-038 — how tensor shapes reach the cost model (the collector captures node output shapes).
