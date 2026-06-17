"""Generate the hero-report demo scene — one capture that shows both of compile-lens's headline
stories in a single HTML report.

The model is a stack of three ``GEMM-Residual-RMSNorm-GEMM`` blocks (the shape real transformer FFNs
present) followed by a small affine tail ``(x - bias) / scale``:

- **Pillar — regression governance.** The ``head`` variant flips the tail's centering subtraction to
  ``(bias - x) / scale``. That is a *silent* sign error — it raises no exception, it just produces
  wrong numbers. ``cl diff --base hero_base --head hero_head`` recovers it as exactly **one
  ``modified`` node** (the ``sub``) at 100% match coverage: "this PR changed exactly one thing, here
  it is, and we are sure."
- **Crown jewel — fusion opportunities.** Each of the three blocks is a CODA ``Pattern A`` the
  Inductor leaves unfused; ``cl fusion-detect`` (and the report's fusion section) flags all three,
  with an analytical HBM-traffic estimate (a memory-bound upper bound, not a measured speedup).

Run it, then render the report::

    python examples/hero_scene.py                       # writes examples/hero_{base,head}.cls.json
    cl session report examples/hero_head.cls.json \\
        --base examples/hero_base.cls.json --output hero.html
    cl scrub hero.html                                  # already share-safe; this just confirms it
    # open hero.html

The capture runs on CPU and needs only PyTorch — no GPU. Session metadata is pinned so the artifacts
are deterministic; the committed copies are scrubbed to ``public-safe`` so they carry no host or
command.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from compile_lens.collectors.artifact import CompileArtifactCollector

HERE = Path(__file__).resolve().parent

# Pinned so two runs produce byte-identical artifacts (the diff/report stay reproducible).
_SESSION = {
    "session_id": "00000000-0000-4000-8000-00000000he50",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.12.0",
}


class Block(nn.Module):
    """One ``GEMM-Residual-RMSNorm-GEMM`` block: ``w1(rmsnorm(w0(x) + resid))``.

    ``w0`` widens d_model→d_ff (GEMM1), a full-shape residual is added, RMSNorm normalizes, and
    ``w1`` projects d_ff→d_model (GEMM2) so blocks chain. This is the single-consumer chain CODA
    Pattern A fuses — the per-row inverse-RMS scalar can be folded into GEMM2's epilogue so the
    normalized tensor never lands in HBM.
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


class Net(nn.Module):
    """Three blocks plus an affine tail. ``swap`` flips the tail's subtraction operands — a silent
    sign bug — so a base/head pair differs by exactly that one non-commutative operation."""

    def __init__(self, n_blocks: int, d_model: int, d_ff: int, rows: int, swap: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(d_model, d_ff, rows) for _ in range(n_blocks)])
        self.bias = nn.Parameter(torch.randn(rows, d_model))
        self.scale = nn.Parameter(torch.ones(rows, d_model))
        self.swap = swap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        # The bug: a refactor flips the centering subtraction. Non-commutative → silently wrong.
        return (self.bias - x) / self.scale if self.swap else (x - self.bias) / self.scale


def main() -> None:
    for name, swap in (("hero_base", False), ("hero_head", True)):
        torch._dynamo.reset()
        # Seed identically before each model so base and head share the same weights — then the
        # only difference the diff finds is the operand swap, and the artifacts are deterministic.
        torch.manual_seed(0)
        model = Net(3, 64, 128, 8, swap)
        torch.manual_seed(0)
        collector = CompileArtifactCollector(HERE / f"{name}.cls.json", **_SESSION)
        collector.capture(model, torch.randn(8, 64))
        path = collector.finalize()
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
