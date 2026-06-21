"""End-to-end integration test for Tool 6 (CODA fusion-opportunity detector).

Drives the full pipeline across the language boundary, on a *real* `torch.compile` capture:

    nn.Module (3 GEMM-Residual-RMSNorm-GEMM blocks)
        --(Python collector: torch.compile capture)-->  .cls.json
    .cls.json
        --(Rust `cl fusion-detect --format json`)-->  fusion_opportunities[]

then asserts the detector found the three Pattern A opportunities, with shapes derived and a cost
estimated. This validates the matcher + shape-wiring + cost model + renderer on a graph torch
actually produced — not a hand-built fixture. In particular it pins that the detector fires on the
*decomposed* RMSNorm subgraph (`pow→mean→rsqrt→mul`) that `nn.RMSNorm` lowers to — the form real
models present (a native `aten.rms_norm` node is the exception, not the rule).

Needs torch, so it `importorskip`s — it runs wherever the capture path's torch is installed (the
same gate the collector's capture unit tests use) and skips cleanly where it isn't. The aten capture
runs on CPU with fake tensors, so no GPU is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (only reachable once torch imported)

from compile_lens.collectors.artifact import CompileArtifactCollector  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

_SESSION_KW = {
    "session_id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.6.0",
}

SUGGESTED_FUSION = "fold per-row 1/rms into the second GEMM epilogue"


class _Block(nn.Module):
    """One GEMM-Residual-RMSNorm-GEMM block: ``w1(rmsnorm(w0(x) + resid))``.

    ``w0`` widens d_model→d_ff (GEMM1), a fixed residual is added, RMSNorm normalizes, and ``w1``
    projects d_ff→d_model (GEMM2) so blocks chain. The residual is a full-shape buffer (not a
    broadcast), matching the single-consumer chain Pattern A requires.
    """

    def __init__(self, d_model: int, d_ff: int, rows: int) -> None:
        super().__init__()
        self.w0 = nn.Linear(d_model, d_ff, bias=True)
        self.norm = nn.RMSNorm(d_ff)
        self.w1 = nn.Linear(d_ff, d_model, bias=False)
        self.register_buffer("resid", torch.randn(rows, d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w0(x)
        h = h + self.resid
        h = self.norm(h)
        return self.w1(h)


class _Net(nn.Module):
    """A stack of blocks → one fusion opportunity per block."""

    def __init__(self, n_blocks: int, d_model: int, d_ff: int, rows: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(d_model, d_ff, rows) for _ in range(n_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _capture(out: Path, n_blocks: int, d_model: int, d_ff: int, rows: int) -> None:
    torch.manual_seed(0)
    torch._dynamo.reset()
    collector = CompileArtifactCollector(out, **_SESSION_KW)  # type: ignore[arg-type]
    collector.capture(_Net(n_blocks, d_model, d_ff, rows), torch.randn(rows, d_model))
    collector.finalize()


def _fusion_detect_json(session: Path, *extra: str) -> dict:
    """Run `cl fusion-detect <session> --format json [extra...]` and parse stdout."""
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
            "fusion-detect",
            str(session),
            "--format",
            "json",
            *extra,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_detects_three_pattern_a_opportunities_with_cost(tmp_path: Path) -> None:
    out = tmp_path / "transformer.cls.json"
    _capture(out, n_blocks=3, d_model=64, d_ff=128, rows=32)

    report = _fusion_detect_json(out)
    opps = report["fusion_opportunities"]

    # AC #3: at least three opportunities on a LLaMA-style multi-block graph.
    assert len(opps) >= 3, (
        f"expected >=3 opportunities, got {len(opps)}:\n{json.dumps(opps, indent=2)}"
    )

    for opp in opps:
        assert opp["pattern_id"] == "A"
        assert opp["suggested_fusion"] == SUGGESTED_FUSION
        # Real nn.RMSNorm lowers to the decomposed subgraph → medium; allow high if a torch
        # version emits a native aten.rms_norm instead.
        assert opp["confidence"] in {"medium", "high"}
        # Shapes were captured, so the cost model ran: GEMM dims + a memory-bound speedup > 1.
        shape = opp["shape"]
        assert shape["m"] == 32 and shape["k0"] == 64 and shape["n"] == 128 and shape["k1"] == 64
        assert opp["baseline_hbm_bytes"] > opp["fused_hbm_bytes"] > 0
        assert opp["estimated_speedup"] > 1.0


def test_min_speedup_floor_can_filter_everything(tmp_path: Path) -> None:
    """A floor above any achievable memory-bound ratio drops every opportunity — exit 0, empty."""
    out = tmp_path / "transformer.cls.json"
    _capture(out, n_blocks=3, d_model=64, d_ff=128, rows=32)

    report = _fusion_detect_json(out, "--min-speedup", "1000")
    assert report["fusion_opportunities"] == []
