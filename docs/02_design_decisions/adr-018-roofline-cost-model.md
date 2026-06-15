# ADR-018: Roofline cost model — layer separation, rank-correlation gating, spill softening

- **Status**: Accepted
- **Date**: 2026-06 (Tool 5 / Phase 6; this record written 2026-06-15 to back the in-code and tool-doc citations of ADR-018 that predated the file)
- **Deciders**: project maintainer
- **Related**: Tool 5 (`kernel-roofline`); the `cls-roofline` crate (`roofline.rs` / `predictor.rs` / `pruning.rs`); `docs/03_tools/kernel_roofline.md`.

## Context

Tool 5 triages a Triton kernel against the roofline and prunes an autotune grid: for each kernel it computes a theoretical lower bound, inflates it into a ranking predictor, and — when measurements are present — decides how many configs an autotuner must actually measure. Three choices shape how that cost model is *structured* and how it *earns trust*. They were referenced as "ADR-018" throughout the code and the tool page before this file existed; this records them.

## Decision

**1. Three layers, kept separate — never prune on the sound lower bound.**

- **Layer 1 — theoretical lower bound** (Williams roofline): `max(F/C, B/M)`. This is a *sound* floor — no kernel on this GPU can beat it.
- **Layer 2 — empirical predictor**: the lower bound inflated by four corrections (block size, occupancy, launch overhead, register pressure), used for **ranking**.
- **Layer 3 — calibrated pruning**: gated on how well the predicted ranking matches a measured sample.

Layer 1 is a **reference only** and is deliberately *not* used for pruning. Reporting a sound lower bound while simultaneously pruning on a predictor that can exceed it would be self-contradictory — claiming "no kernel beats X µs" while ranking a config as faster than X. So the sound bound *informs* and the predictor *ranks*, and they never feed the same decision.

**2. Spearman rank correlation, not Pearson, for the pruning gate.**

Autotuning is a **ranking** task — keep the top-K fastest. A high Pearson correlation can coexist with a wrong top-K order (the predicted-vs-measured relationship is linear-ish overall, but the fastest few are mis-ordered). Rank correlation measures the agreement that actually matters for pruning. The gate, on the Spearman correlation between predicted and measured runtimes:

| Rank correlation | Tier | Action |
|---|---|---|
| ≥ 0.8 | aggressive | measure only the predicted top-K |
| 0.5 – 0.8 | moderate | measure the top half |
| < 0.5 or undefined | full-sweep fallback | measure everything |

The undefined case (e.g. a memory-bound kernel whose configs all predict identically) falls back to a full sweep rather than pruning on a ranking the model does not have. The schema field is `threshold_rank_correlation`.

**3. Register-spill penalty softened.**

An earlier draft modeled a register spill as roughly doubling bandwidth (the "biggest factor on H100"). That overstates it: a spill goes to **local memory**, which is usually L1/L2-cached and reaches HBM only on a miss — so a spill is a *modest* amplifier on register-bound kernels, not a 2× bandwidth blow-up. The penalty is capped well below 2× (a fraction of the lower bound, `SPILL_PENALTY_CAP`).

## Alternatives considered

- **Prune on the Layer 1 sound bound (rejected).** It is a floor, not a predictor; identical for configs that move the same bytes, so it ranks nothing among them. Using it to prune contradicts reporting it as a sound bound.
- **Pearson gating (rejected).** Pearson measures linear fit; a ranking task needs a *rank* statistic. A strong linear fit with a mis-ordered fastest-few would prune away the real winner.
- **Spill as a ~2× bandwidth amplifier (rejected).** Overstates HBM impact; spilled registers land in cached local memory, not unconditionally in HBM.

## Consequences

- The model earns trust through Layer 3's **measured** rank correlation, not through the Layer 2 coefficients being exact — they are documented, calibration-tunable defaults.
- Pruning is only ever as aggressive as the correlation earns; a kernel the model cannot rank costs nothing in safety, only the measurement time it would have taken anyway.
- The separation makes each layer independently checkable: Layer 1 against the hardware spec, Layer 2's monotonic ranking against measurements, Layer 3's gate against the correlation thresholds.
