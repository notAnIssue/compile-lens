# Real-model compile-diff pairs

Captured FX-graph pairs from real PyTorch small models, for the match-coverage benchmark
(`crates/cls-wl-diff/tests/coverage_benchmark.rs`). These are the **real** counterpart to the
synthetic fixtures one directory up: produced by running `CompileArtifactCollector` over an
actual `torch.compile`, not hand-authored. Each is a small model and a minor variant.

Each `<name>/base.cls.json` and `<name>/head.cls.json` is the aten-normalized graph the collector
captured (op types like `aten.addmm.default`), so the benchmark exercises the matcher on the kind
of graph it will see in production rather than only the controlled synthetic cases.

| Pair | Change | Notes |
|---|---|---|
| `depth` | add one `Linear`+`ReLU` block | |
| `activation` | `ReLU` → `GELU` | |
| `width` | hidden dim 8 → 16 | structure identical, only shapes change |
| `residual` | add an input residual | |
| `bias` | `Linear(bias=True)` → `bias=False` | the largest change here: it turns the linear's `addmm` into `mm` and drops the bias input, which ripples into the `relu` (its input op changed), so its coverage is the lowest of the set — a faithful measurement, which is why the benchmark asserts the **median** (robust to one hard pair), not the minimum |

The benchmark asserts the median `match_coverage` clears 0.70. A median below that is a release
blocker (design D7), not a threshold to lower.
