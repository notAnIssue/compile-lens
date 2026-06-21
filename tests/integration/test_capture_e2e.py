"""End-to-end test for ``cl.capture()`` — the active one-call capture.

Drives the orchestrator over a small model and asserts that one call assembles every CPU-capturable
section into one artifact: compiled graphs, per-iteration cache data, the recompile log under shape
variation, a lint candidate from the source, and (when a real divergence exists) the localized
layer. Needs torch, so it ``importorskip``s and runs on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import compile_lens as cl  # noqa: E402

_SESSION = {
    "session_id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.6.0",
}


class _Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w0 = nn.Linear(d_model, d_ff)
        self.norm = nn.RMSNorm(d_ff)
        self.w1 = nn.Linear(d_ff, d_model, bias=False)
        self.resid = nn.Parameter(torch.randn(d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(self.norm(self.w0(x) + self.resid))


class _Net(nn.Module):
    def __init__(self, *, swap: bool) -> None:
        super().__init__()
        self.block = _Block(16, 32)
        self.bias = nn.Parameter(torch.randn(16))
        self.swap = swap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.bias - x if self.swap else x - self.bias


_LINT_SRC = (
    "import torch\n"
    "def apply_mask(x, mask):\n"
    "    flat = x.view(-1)\n"
    "    flat[mask] = 0.0\n"
    "    return flat\n"
)


def _build(*, swap: bool) -> _Net:
    torch.manual_seed(0)
    return _Net(swap=swap)


def test_capture_assembles_every_cpu_section(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "model_src.py"
    src.write_text(_LINT_SRC)

    def buggy_pass(graph: object) -> None:
        for node in graph.nodes:  # type: ignore[attr-defined]
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.sub.Tensor
                break
        graph.lint()  # type: ignore[attr-defined]

    from torch._inductor import config as inductor_config

    inductor_config.post_grad_custom_post_pass = buggy_pass
    # The recompile probe captures the stderr fd; pytest redirects it, so torch's recompile log
    # would bypass the tee. `capfd.disabled()` restores the real fds for the capture (a real user's
    # script is always in that condition); reset the recompiles log first so its handler re-binds.
    try:
        with capfd.disabled():
            torch._logging.set_logs(recompiles=False)
            result = cl.capture(
                _build(swap=True),
                torch.randn(4, 16),
                base=_build(swap=False),
                vary_inputs=[torch.randn(rows, 16) for rows in (4, 8, 16)],
                source=src,
                check_divergence=True,
                output_dir=tmp_path / "out",
                **_SESSION,
            )
    finally:
        inductor_config.post_grad_custom_post_pass = None

    head = json.loads(result.artifact_path.read_text())
    assert head["compiled_graphs"], "graph capture (fusion / diff) must be present"
    assert head["iterations"], "per-iteration cache data must be present"
    assert head["recompilations"], "shape variation must have produced recompiles"
    assert [f["pattern_category"] for f in head["lint_findings"]] == ["in_place_op_on_alias"]
    assert head["divergences"][0]["first_divergent_layer"] is not None

    assert result.base_path is not None
    base = json.loads(result.base_path.read_text())
    assert base["compiled_graphs"], "base graph must be captured for the IR diff"


def test_capture_reports_no_divergence_for_a_clean_model(tmp_path: Path) -> None:
    """Without a miscompiling pass, eager and compiled agree — divergences must be absent, not a
    fabricated finding."""
    result = cl.capture(
        _build(swap=False),
        torch.randn(4, 16),
        check_divergence=True,
        output_dir=tmp_path / "out",
        **_SESSION,
    )
    head = json.loads(result.artifact_path.read_text())
    assert "divergences" not in head or head["divergences"] == []
