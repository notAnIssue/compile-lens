"""End-to-end integration tests for Tool 5 (kernel roofline triage).

Drives the analysis the way the autotune harness will: a committed `.cls.json` of kernel configs
(features + measured runtimes) is fed to the built `cl kernel-roofline` binary, which computes each
config's roofline prediction (Layer 1 lower bound + Layer 2 empirical predictor) and, from the
measured runtimes, a grid-level calibration and pruning decision (Layer 3). This exercises the same
JSON-file + subprocess boundary the harness uses (`--format json`), per the design doc ADR-006 — the
boundary is a file, not FFI.

The fixture is monotone by construction: block size shrinks / register pressure rises across the
four configs so the predictor ranks them the same way the measured runtimes do, giving a perfect
rank correlation and therefore aggressive pruning.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "roofline" / "sample.cls.json"


def _build_cl() -> None:
    subprocess.run(
        ["cargo", "build", "--quiet", "--package", "cls-cli", "--bin", "cl"],
        cwd=REPO_ROOT,
        check=True,
    )


def _run(*args: str) -> str:
    """Run `cl kernel-roofline <fixture> <args...>` and return stdout."""
    _build_cl()
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--package",
            "cls-cli",
            "--bin",
            "cl",
            "--",
            "kernel-roofline",
            str(FIXTURE),
            *args,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_list_enumerates_kernels() -> None:
    ids = [line.split("\t")[0] for line in _run("--list").splitlines() if line.strip()]
    assert ids == ["cfg_bs256_r32", "cfg_bs64_r32", "cfg_bs32_r32", "cfg_bs256_r64"]


def test_json_report_has_predictions_calibration_and_pruning() -> None:
    report = json.loads(_run("--format", "json", "--top-k", "2"))
    assert report["gpu"] == "A100-SXM-80GB"

    # One prediction per kernel, each with the Layer-2 empirical predictor filled.
    preds = report["predictions"]
    assert len(preds) == 4
    assert all(p["empirical_predictor_us"] > 0 for p in preds)

    # The predictor ranks the configs the way the measured runtimes do (monotone fixture).
    predicted = [p["empirical_predictor_us"] for p in preds]
    assert predicted == sorted(predicted)

    # Perfect rank correlation -> aggressive pruning measures only the top-k of the grid.
    pruning = report["pruning_decision"]
    assert pruning["mode"] == "aggressive"
    assert pruning["configs_total"] == 4
    assert pruning["configs_measured"] == 2
    assert report["calibration"]["spearman_correlation"] == 1.0
    assert report["calibration"]["calibration_verdict"] == "reliable"


def test_kernel_filter_narrows_the_report() -> None:
    report = json.loads(_run("--format", "json", "--kernel", "bs64"))
    ids = [p["kernel_id"] for p in report["predictions"]]
    assert ids == ["cfg_bs64_r32"]
