"""Tests for the autotune harness (Tool 5).

The harness's job is orchestration: predict the grid (Rust, via subprocess), measure a spread
calibration sample, read back the prune tier, then measure only what the tier allows. These tests
stub the three I/O seams — ``_predict`` (the Rust prediction), ``_measure`` (GPU timing), and
``_calibrate`` (the Rust tier decision) — so the ranking, calibration-sample selection, prune-mode
application, pruning-ratio math, and winner selection are exercised deterministically without a GPU
or the `cl` binary. A real GPU sweep is a separate smoke (``test_autotune_harness_gpu.py``).
"""

from __future__ import annotations

import pytest

from compile_lens.kernels.autotune_harness import AutotuneHarness, _expand_grid, _verdict


def _harness(
    predicted: list[float],
    measured: dict[int, float],
    mode: str,
    *,
    rho: float | None = 1.0,
    keep_top_k: int = 4,
    calibration_size: int = 2,
) -> AutotuneHarness:
    """A harness over ``len(predicted)`` configs (config i is ``{"id": i}``) with the three GPU/Rust
    seams stubbed to the given prediction, measurement map, and prune tier."""
    n = len(predicted)
    h = AutotuneHarness(
        {"id": list(range(n))},
        features_for=lambda c: {"flops": 1e9, "bytes_loaded": 1e8, "bytes_stored": 0.0},
        run_for=lambda c: lambda: None,
        gpu="A100-SXM-80GB",
        keep_top_k=keep_top_k,
        calibration_size=calibration_size,
    )
    h._predict = lambda: list(predicted)  # type: ignore[method-assign]
    h._measure = lambda config: measured[config["id"]]  # type: ignore[method-assign]
    h._calibrate = lambda cal_idx, pred, meas: (mode, rho)  # type: ignore[method-assign]
    return h


class TestGridExpansion:
    def test_cartesian_product(self) -> None:
        configs = _expand_grid({"BLOCK": [64, 128], "warps": [4, 8]})
        assert len(configs) == 4
        assert {"BLOCK": 64, "warps": 4} in configs
        assert {"BLOCK": 128, "warps": 8} in configs

    def test_list_passthrough(self) -> None:
        lst = [{"a": 1}, {"a": 2}]
        assert _expand_grid(lst) == lst


class TestSweep:
    def test_aggressive_prunes_to_top_k_and_hits_4x(self) -> None:
        # 24 configs, predicted == measured ranking (config 0 cheapest), tier aggressive.
        n = 24
        predicted = [float(i) for i in range(n)]
        measured = {i: 100.0 + i for i in range(n)}
        report = _harness(
            predicted, measured, "aggressive", keep_top_k=4, calibration_size=2
        ).sweep()

        assert report.prune_mode == "aggressive"
        assert report.calibration_verdict == "reliable"
        assert report.rank_correlation == 1.0
        # top-4 by predicted + 2 spread calibration configs = 6 measured of 24 -> 4x.
        assert report.n_measured == 6
        assert report.pruning_ratio == pytest.approx(4.0)
        assert report.best_config == {"id": 0}
        assert report.best_us == 100.0

    def test_disabled_measures_everything(self) -> None:
        n = 10
        predicted = [float(i) for i in range(n)]
        measured = {i: 100.0 + i for i in range(n)}
        report = _harness(
            predicted, measured, "disabled_fallback_full_sweep", rho=0.2, keep_top_k=2
        ).sweep()

        assert report.prune_mode == "disabled_fallback_full_sweep"
        assert report.calibration_verdict == "unreliable"
        assert report.n_measured == n
        assert report.pruning_ratio == pytest.approx(1.0)

    def test_moderate_measures_top_half(self) -> None:
        n = 10
        predicted = [float(i) for i in range(n)]
        measured = {i: 100.0 + i for i in range(n)}
        report = _harness(predicted, measured, "moderate", rho=0.6, keep_top_k=2).sweep()

        assert report.prune_mode == "moderate"
        assert report.calibration_verdict == "partial"
        # top half (5) + calibration {3, 7} -> {0,1,2,3,4,7} = 6 measured.
        assert report.n_measured == 6

    def test_winner_is_lowest_measured_not_lowest_predicted(self) -> None:
        # The predictor ranks config 0 best, but config 2 (also measured, in the top-k) is faster.
        n = 8
        predicted = [float(i) for i in range(n)]
        measured = {i: 100.0 + i for i in range(n)}
        measured[2] = 50.0
        report = _harness(predicted, measured, "aggressive", keep_top_k=4).sweep()
        assert report.best_config == {"id": 2}
        assert report.best_us == 50.0

    def test_rows_cover_every_config_measured_only_for_chosen(self) -> None:
        n = 12
        predicted = [float(i) for i in range(n)]
        measured = {i: 100.0 + i for i in range(n)}
        report = _harness(
            predicted, measured, "aggressive", keep_top_k=3, calibration_size=2
        ).sweep()
        assert len(report.rows) == n  # a row per config
        measured_rows = [r for r in report.rows if r.measured_us is not None]
        assert len(measured_rows) == report.n_measured
        # every row has a prediction
        assert all(r.predicted_us is not None for r in report.rows)

    def test_empty_grid_raises(self) -> None:
        h = AutotuneHarness([], features_for=lambda c: {}, run_for=lambda c: lambda: None)
        with pytest.raises(ValueError, match="empty"):
            h.sweep()


class TestReportOutputs:
    def test_to_csv_round_trips(self, tmp_path) -> None:
        import csv

        n = 6
        predicted = [float(i) for i in range(n)]
        measured = {i: 100.0 + i for i in range(n)}
        report = _harness(
            predicted, measured, "aggressive", keep_top_k=2, calibration_size=2
        ).sweep()
        out = report.to_csv(tmp_path / "sweep.csv")

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n
        assert "predicted_us" in rows[0]
        assert "measured_us" in rows[0]
        assert "id" in rows[0]  # the config column


def test_verdict_maps_tiers() -> None:
    assert _verdict("aggressive") == "reliable"
    assert _verdict("moderate") == "partial"
    assert _verdict("disabled_fallback_full_sweep") == "unreliable"
