"""``AutotuneHarness`` — predicted-vs-measured autotune sweep with calibrated pruning (Tool 5).

Autotuning a Triton kernel means trying many configs and keeping the fastest. The naive way measures
every config; this harness uses the roofline cost model to *predict* the ranking, measures only a
small calibration sample to check the prediction is trustworthy, and then measures only the
predicted top-K — so a 24-config grid can cost a handful of measurements instead of 24.

The cost model itself lives once, in Rust (`cls-roofline`). The harness drives it out of process:
it writes the configs to a `.cls.json` and shells out to `cl kernel-roofline --format json`
(Python↔Rust exchange is a JSON file over a subprocess, never an in-process binding). Spearman
correlation and the prune-tier thresholds stay in Rust too; the harness reads the
tier back and only *applies* it to the grid. So the harness is pure orchestration.

The two GPU-touching steps are injected as callbacks so the orchestration is testable without a GPU:

  * ``features_for(config) -> dict`` — the kernel features the cost model reads (flops / bytes /
    block_size / num_regs / n_spills). Real use compiles the Triton kernel with the config and reads
    them off the compiled kernel (the Tool 5 metadata collector); tests pass a stub.
  * ``run_for(config) -> Callable[[], None]`` — a zero-arg closure that runs the kernel with the
    config, for timing via the self-timed path (the proton adapter).

A real GPU sweep wires these to Triton + torch; the unit tests stub the three I/O seams
(``_predict`` / ``_measure`` / ``_calibrate``) and exercise the ranking, calibration, prune-mode
application, and pruning-ratio math deterministically.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compile_lens.kernels import proton_adapter

_log = structlog.get_logger("compile_lens.kernels.autotune_harness")

Config = Mapping[str, Any]


@dataclass
class SweepRow:
    """One config's place in the sweep: its prediction, and its measurement if it was measured."""

    config: dict[str, Any]
    predicted_us: float
    measured_us: float | None = None


@dataclass
class SweepReport:
    """A sweep outcome: the winning config, the per-config table, and the pruning achieved."""

    gpu: str
    best_config: dict[str, Any]
    best_us: float
    rows: list[SweepRow]
    n_configs: int
    n_measured: int
    pruning_ratio: float
    prune_mode: str
    rank_correlation: float | None
    calibration_verdict: str | None = field(default=None)

    def to_csv(self, path: str | Path) -> Path:
        """Write the per-config table (config columns + predicted/measured) as CSV."""
        import csv  # noqa: PLC0415 — stdlib, lazy to keep import cost off the hot path

        out = Path(path)
        config_keys = sorted({k for row in self.rows for k in row.config})
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([*config_keys, "predicted_us", "measured_us"])
            for row in self.rows:
                writer.writerow(
                    [
                        *(row.config.get(k, "") for k in config_keys),
                        row.predicted_us,
                        row.measured_us,
                    ]
                )
        return out

    def plot_predicted_vs_measured(self, path: str | Path) -> Path | None:
        """Scatter predicted vs measured. Skips if matplotlib is absent."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except Exception:  # pragma: no cover — plotting is optional
            _log.warning("autotune_harness.plot_skipped", reason="matplotlib unavailable")
            return None

        pairs = [(r.predicted_us, r.measured_us) for r in self.rows if r.measured_us is not None]
        if not pairs:  # pragma: no cover
            return None
        xs, ys = zip(*pairs, strict=True)
        fig, ax = plt.subplots()
        ax.scatter(xs, ys)
        ax.set_xlabel("predicted µs")
        ax.set_ylabel("measured µs")
        ax.set_title(f"Predicted vs measured ({self.gpu}, ρ={self.rank_correlation})")
        out = Path(path)
        fig.savefig(out)
        plt.close(fig)
        return out


def _resolve_cl_bin() -> str:
    """The `cl` binary to drive: ``$CL_BIN`` if set, else ``cl`` on PATH."""
    return os.environ.get("CL_BIN", "cl")


class AutotuneHarness:
    """Predicted-vs-measured autotune sweep with calibrated pruning.

    Args:
        config_grid: a mapping of knob -> list of values; the sweep is the Cartesian product. (A
            pre-expanded list of config dicts is also accepted.)
        features_for: ``config -> kernel-features dict`` the cost model reads.
        run_for: ``config -> zero-arg callable`` that runs the kernel once, for timing.
        gpu: GPU spec name passed to ``cl kernel-roofline --gpu`` for the prediction.
        keep_top_k: under aggressive pruning, measure this many predicted-best configs.
        calibration_size: how many configs (spread across the predicted ranking) to measure for the
            trust check. Defaults to a small spread sample.
        iterations / warmup: self-timed measurement parameters.
        cl_bin: override the `cl` binary path (defaults to ``$CL_BIN`` or ``cl``).
    """

    def __init__(
        self,
        config_grid: Mapping[str, Sequence[Any]] | Sequence[Config],
        features_for: Callable[[Config], Mapping[str, Any]],
        run_for: Callable[[Config], Callable[[], Any]],
        *,
        gpu: str = "A100-SXM-80GB",
        keep_top_k: int = 5,
        calibration_size: int | None = None,
        iterations: int = 30,
        warmup: int = 10,
        cl_bin: str | None = None,
    ) -> None:
        self.configs: list[dict[str, Any]] = _expand_grid(config_grid)
        self.features_for = features_for
        self.run_for = run_for
        self.gpu = gpu
        self.keep_top_k = keep_top_k
        self.calibration_size = calibration_size
        self.iterations = iterations
        self.warmup = warmup
        self.cl_bin = cl_bin or _resolve_cl_bin()

    # ── the sweep ────────────────────────────────────────────────────────────────────────
    def sweep(self) -> SweepReport:
        """Predict the grid, calibrate on a sample, then measure only what the prune tier allows."""
        n = len(self.configs)
        if n == 0:
            raise ValueError("config grid is empty")

        predicted = self._predict()  # one µs prediction per config, in self.configs order
        order = sorted(range(n), key=lambda i: predicted[i])  # cheapest predicted first

        # Measure a spread calibration sample, ask Rust for the prune tier (and ρ) it implies.
        cal_idx = self._calibration_indices(order)
        measured: dict[int, float] = {i: self._measure(self.configs[i]) for i in cal_idx}
        prune_mode, rank_corr = self._calibrate(cal_idx, predicted, measured)

        # Apply the tier to the full grid, measuring only what it allows (calibration counts).
        for i in self._measure_set(order, prune_mode):
            if i not in measured:
                measured[i] = self._measure(self.configs[i])

        best_i = min(measured, key=lambda i: measured[i])
        rows = [
            SweepRow(
                config=dict(self.configs[i]), predicted_us=predicted[i], measured_us=measured.get(i)
            )
            for i in range(n)
        ]
        return SweepReport(
            gpu=self.gpu,
            best_config=dict(self.configs[best_i]),
            best_us=measured[best_i],
            rows=rows,
            n_configs=n,
            n_measured=len(measured),
            pruning_ratio=n / len(measured),
            prune_mode=prune_mode,
            rank_correlation=rank_corr,
            calibration_verdict=_verdict(prune_mode),
        )

    def _calibration_indices(self, order: list[int]) -> list[int]:
        """Pick configs spread across the predicted ranking (not just the top) so the correlation
        check spans the range. Defaults to a small sample; never fewer than two (a correlation needs
        two points), never more than the grid."""
        n = len(order)
        size = self.calibration_size if self.calibration_size is not None else max(2, n // 8)
        size = max(2, min(size, n))
        # Evenly spaced interior positions of the ranking.
        positions = sorted({min(n - 1, round((k + 1) * n / (size + 1))) for k in range(size)})
        return [order[p] for p in positions]

    def _measure_set(self, order: list[int], prune_mode: str) -> list[int]:
        """Which configs to measure under the tier: top-K, top half, or all."""
        n = len(order)
        if prune_mode == "aggressive":
            return order[: self.keep_top_k]
        if prune_mode == "moderate":
            return order[: math.ceil(n / 2)]
        return order  # disabled_fallback_full_sweep — no trust, measure everything

    # ── I/O seams (stubbed in unit tests; real on a GPU) ──────────────────────────────────
    def _predict(self) -> list[float]:
        """Predicted empirical-predictor µs per config (self.configs order), via the Rust model.

        Writes a features-only `.cls.json` and shells out to `cl kernel-roofline --format json`.
        """
        kernels = [
            {
                "kernel_id": f"cfg_{i}",
                "name": "autotune_config",
                "features": dict(self.features_for(c)),
            }
            for i, c in enumerate(self.configs)
        ]
        report = self._run_kernel_roofline(kernels)
        by_id = {
            p["kernel_id"]: p.get("empirical_predictor_us") for p in report.get("predictions", [])
        }
        missing = [i for i in range(len(self.configs)) if by_id.get(f"cfg_{i}") is None]
        if missing:
            raise RuntimeError(f"cost model returned no prediction for configs {missing}")
        return [float(by_id[f"cfg_{i}"]) for i in range(len(self.configs))]

    def _measure(self, config: Config) -> float:
        """Measured mean µs for one config, self-timed via the proton adapter (needs a GPU)."""
        measurement = proton_adapter.measure(
            fn=self.run_for(config), iterations=self.iterations, warmup=self.warmup
        )
        if measurement is None or measurement.mean_us is None:
            raise RuntimeError(
                "measurement unavailable (no CUDA?) — cannot autotune without timing"
            )
        return measurement.mean_us

    def _calibrate(
        self, cal_idx: list[int], predicted: list[float], measured: dict[int, float]
    ) -> tuple[str, float | None]:
        """Shell out to the Rust model on the measured calibration sample; return its prune tier and
        Spearman ρ. The tier depends only on ρ, so it is the grid-wide decision even though it is
        computed over the sample."""
        kernels = [
            {
                "kernel_id": f"cfg_{i}",
                "name": "autotune_config",
                "features": dict(self.features_for(self.configs[i])),
                "measurements": {"mean_us": measured[i], "source": "self_timed"},
            }
            for i in cal_idx
        ]
        report = self._run_kernel_roofline(kernels)
        pruning = report.get("pruning_decision") or {}
        calibration = report.get("calibration") or {}
        mode = pruning.get("mode", "disabled_fallback_full_sweep")
        return mode, calibration.get("spearman_correlation")

    def _run_kernel_roofline(self, kernels: list[dict[str, Any]]) -> dict[str, Any]:
        """Write the kernels to a temp `.cls.json` and run `cl kernel-roofline --format json`."""
        artifact = {
            "schema_version": "0.5.0",
            "session": {
                "id": "00000000-0000-4000-8000-000000000000",
                "timestamp": "2026-01-01T00:00:00Z",
                "torch_version": "0.0.0",
                "redaction_policy": "default-strict",
            },
            "kernels": kernels,
        }
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "sweep.cls.json"
            session.write_text(json.dumps(artifact))
            proc = subprocess.run(
                [
                    self.cl_bin,
                    "kernel-roofline",
                    str(session),
                    "--gpu",
                    self.gpu,
                    "--top-k",
                    str(self.keep_top_k),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            # Surface the analyzer's typed error (e.g. an unknown --gpu) rather than an opaque
            # CalledProcessError with just an exit code.
            raise RuntimeError(
                f"`{self.cl_bin} kernel-roofline --gpu {self.gpu}` failed "
                f"(exit {proc.returncode}): {proc.stderr.strip()}"
            )
        report: dict[str, Any] = json.loads(proc.stdout)
        return report


def _expand_grid(grid: Mapping[str, Sequence[Any]] | Sequence[Config]) -> list[dict[str, Any]]:
    """A grid mapping (knob -> values) becomes the Cartesian product of configs; a list of config
    dicts passes through."""
    if isinstance(grid, Mapping):
        keys = list(grid)
        return [
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(grid[k] for k in keys))
        ]
    return [dict(c) for c in grid]


def _verdict(prune_mode: str) -> str:
    """Human label mirroring the cost model's calibration verdict."""
    return {
        "aggressive": "reliable",
        "moderate": "partial",
    }.get(prune_mode, "unreliable")
