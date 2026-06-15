"""Subprocess-boundary tests for the Tool 5 autotune harness.

``AutotuneHarness`` predicts a config grid and calibrates a prune tier by shelling out to the built
``cl kernel-roofline`` binary — Python writes a ``.cls.json``, Rust analyzes it, Python reads the
JSON back. The unit tests in ``python/tests/kernels/test_autotune_harness.py`` stub that seam to
test the orchestration; these tests exercise the seam *for real*:

  * the real-binary tests build ``cl`` and drive ``_predict`` / ``_calibrate`` / ``sweep`` through
    a real subprocess (only the GPU measurement is stubbed — the one step that needs a GPU);
  * the fake-binary tests point ``cl_bin`` at a stand-in that fails or returns nothing, covering the
    two ``RuntimeError`` paths the harness raises when the subprocess misbehaves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from compile_lens.kernels.autotune_harness import AutotuneHarness

REPO_ROOT = Path(__file__).resolve().parents[2]

# A monotone grid: larger ``scale`` => more flops/bytes => higher predicted µs. Distinct, ordered
# predictions make the ranking (and a perfectly rank-correlated calibration) deterministic.
GRID = {"scale": [1, 2, 3, 4]}


def _features(config: Any) -> dict[str, Any]:
    scale = config["scale"]
    return {
        "flops": 12_000_000_000 * scale,
        "bytes_loaded": 840_000_000 * scale,
        "bytes_stored": 420_000_000 * scale,
        "block_size": 256,
        "num_regs": 32,
        "n_spills": 0,
    }


def _fake_measure(config: Any) -> float:
    """A GPU-free stand-in for ``_measure``: monotone in scale, so the measured ranking matches the
    predicted one (=> Spearman ρ = 1 => aggressive pruning), deterministically."""
    return float(_features(config)["bytes_loaded"])


def _harness(cl_bin: str, **kw: Any) -> AutotuneHarness:
    return AutotuneHarness(
        GRID,
        features_for=_features,
        run_for=lambda c: lambda: None,  # never invoked: _measure is stubbed or not reached
        cl_bin=cl_bin,
        **kw,
    )


# ── real cl binary ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cl_bin() -> str:
    """Build ``cl`` once and return the binary path the harness shells out to."""
    subprocess.run(
        ["cargo", "build", "--quiet", "--package", "cls-cli", "--bin", "cl"],
        cwd=REPO_ROOT,
        check=True,
    )
    binary = REPO_ROOT / "target" / "debug" / "cl"
    assert binary.exists(), f"cl not built at {binary}"
    return str(binary)


def test_predict_drives_the_real_binary(cl_bin: str) -> None:
    """``_predict`` writes a features ``.cls.json``, runs ``cl kernel-roofline --format json``, and
    reads back one empirical-predictor µs per config, in grid order."""
    predicted = _harness(cl_bin)._predict()
    assert len(predicted) == 4
    assert all(p > 0 for p in predicted)
    # Monotone fixture: more work => higher predicted µs, strictly increasing with scale.
    assert predicted == sorted(predicted)
    assert len(set(predicted)) == 4  # distinct => unambiguous ranking


def test_calibrate_returns_real_tier_and_correlation(cl_bin: str) -> None:
    """``_calibrate`` sends the measured calibration sample to Rust and reads back the Spearman ρ,
    and the prune tier it implies. Measurements rank-correlated with the prediction => ρ = 1 =>
    aggressive."""
    h = _harness(cl_bin)
    predicted = h._predict()
    order = sorted(range(len(predicted)), key=lambda i: predicted[i])
    cal_idx = h._calibration_indices(order)
    # Measured perfectly rank-correlated with predicted (proportional) => ρ = 1.0.
    measured = {i: predicted[i] * 1000.0 for i in cal_idx}
    mode, rho = h._calibrate(cal_idx, predicted, measured)
    assert rho == 1.0
    assert mode == "aggressive"


def test_sweep_through_real_predict_and_calibrate(cl_bin: str) -> None:
    """The full sweep with only the GPU measurement stubbed: real ``_predict`` + real ``_calibrate``
    subprocesses drive the orchestration. ρ = 1 => aggressive => the grid is pruned, not fully
    measured."""
    h = _harness(cl_bin, keep_top_k=2)
    h._measure = _fake_measure  # type: ignore[method-assign]  # the one GPU seam
    report = h.sweep()

    assert report.n_configs == 4
    assert report.prune_mode == "aggressive"
    assert report.rank_correlation == 1.0
    assert report.calibration_verdict == "reliable"
    assert report.n_measured < report.n_configs  # pruning actually skipped measurements
    assert report.pruning_ratio == 4 / report.n_measured
    assert len(report.rows) == 4  # every config is reported, measured or not
    # The winner is the lowest *measured* config; scale=1 has the smallest stubbed measurement and
    # is in the predicted top-k, so it is measured and wins.
    assert report.best_config == {"scale": 1}


# ── fake cl binary: the two RuntimeError paths ──────────────────────────────────────────────────


def _fake_cl(tmp_path: Path, body: str) -> str:
    """A stand-in ``cl`` that ignores its args and runs ``body``. Returns its path."""
    script = tmp_path / "fake_cl"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return str(script)


def test_nonzero_exit_raises_runtime_error(tmp_path: Path) -> None:
    """A non-zero exit from ``cl`` becomes a RuntimeError carrying the exit code — not an opaque
    CalledProcessError with no context."""
    h = _harness(_fake_cl(tmp_path, "exit 3"))
    with pytest.raises(RuntimeError, match=r"failed \(exit 3\)"):
        h._predict()


def test_missing_prediction_raises_runtime_error(tmp_path: Path) -> None:
    """A successful run that returns no prediction for a config is a RuntimeError — the harness
    cannot rank a config it has no prediction for."""
    h = _harness(_fake_cl(tmp_path, "echo '{\"predictions\": []}'"))
    with pytest.raises(RuntimeError, match="no prediction for configs"):
        h._predict()
