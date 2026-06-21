"""Generate the hero-report demo scene: ONE ``cl.capture()`` call over one realistic workload, from
which every CPU-capturable tool fires on a single ``torch.compile`` model.

The model is a three-block transformer FFN stack — each block a ``GEMM → +residual → RMSNorm →
GEMM`` chain (the CODA "Pattern A") — followed by an affine tail ``(x - bias) / scale``. Three
things are wrong with this particular run, the way a real messy PR is wrong in several ways at once:

- **A silent sign regression.** The ``head`` variant flips the tail to ``(bias - x) / scale``. It
  raises nothing; it just computes wrong numbers. The IR diff recovers it as exactly one modified
  node against the ``base``.
- **A miscompiling custom pass.** A custom post-grad "fusion" pass (the kind a team adds for a perf
  win) rewrites an ``add`` into a ``sub`` — so the *compiled* model diverges from eager. Tool 3
  localizes the first layer that disagrees.
- **An in-place-on-alias correctness risk** in a helper, which the static lint scan flags.

Run it across four sequence lengths and it also recompiles; run it repeatedly at one shape and the
cache stays stable. ``cl.capture()`` drives all of that in one call and writes one ``.cls.json``
(plus the base). The report then shows: Recompile (Tool 1), IR Diff (Tool 2a), Cache stability
(Tool 2b, the honest no-bug case), Divergence (Tool 3), Lint (Tool 4), and Fusion (Tool 6).
Roofline (Tool 5) needs a measured GPU kernel profile and renders as such — it is not capturable on
CPU.

Reproduce::

    python examples/hero_scene.py     # writes examples/hero_{base,head}.cls.json (needs torch)
    ./scripts/render_hero.sh          # renders examples/hero.html (no torch needed)

The capture runs on CPU. Session metadata is pinned so the artifacts are deterministic.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import compile_lens as cl

HERE = Path(__file__).resolve().parent

# Pinned so re-capture is deterministic (the committed artifacts stay byte-stable).
_SESSION = {
    "session_id": "00000000-0000-4000-8000-00000000he50",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.12.0",
}


class Block(nn.Module):
    """One ``GEMM → +residual → RMSNorm → GEMM`` block (CODA Pattern A).

    ``w0`` widens d_model→d_ff (GEMM1), a per-feature residual is added (it broadcasts over rows, so
    the block accepts any sequence length and recompiles cleanly on a shape change), RMSNorm
    normalizes, and ``w1`` projects d_ff→d_model (GEMM2). The per-row ``1/rms`` scalar can be folded
    into GEMM2's epilogue so the normalized tensor never lands in HBM — the fusion Tool 6 reports.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w0 = nn.Linear(d_model, d_ff, bias=True)
        self.norm = nn.RMSNorm(d_ff)
        self.w1 = nn.Linear(d_ff, d_model, bias=False)
        self.resid = nn.Parameter(torch.randn(d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w0(x)
        h = h + self.resid
        h = self.norm(h)
        return self.w1(h)


class Net(nn.Module):
    """Three blocks plus an affine tail. ``swap`` flips the tail's subtraction operands — a silent
    sign bug — so a base/head pair differs by exactly that one non-commutative operation."""

    def __init__(self, n_blocks: int, d_model: int, d_ff: int, *, swap: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(d_model, d_ff) for _ in range(n_blocks)])
        self.bias = nn.Parameter(torch.randn(d_model))
        self.scale = nn.Parameter(torch.ones(d_model))
        self.swap = swap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return (self.bias - x) / self.scale if self.swap else (x - self.bias) / self.scale

    def reset_padding(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero padded positions. BUG (Tool 4 — in_place_op_on_alias, Li et al. §3.2.3): it writes
        through ``flat``, a ``view`` that aliases ``x``, so the mutation escapes back into ``x`` —
        unsafe to do inside a region ``torch.compile`` may reorder."""
        flat = x.view(-1)
        flat[mask] = 0.0
        return flat


def _buggy_fusion_pass(graph: Any) -> None:
    """A custom post-grad pass with a sign bug: it rewrites the first ``add`` into a ``sub``.

    Stands in for a real custom inductor pass that miscompiles — the source of the eager-vs-compiled
    divergence Tool 3 localizes. It mutates only the *compiled* graph, so eager stays correct and
    the two disagree."""
    for node in graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor:
            node.target = torch.ops.aten.sub.Tensor
            break
    graph.lint()


def _build(*, swap: bool) -> Net:
    # Seed identically before each model so base and head share weights — then the only difference
    # the diff finds is the operand swap, and the artifacts are deterministic.
    torch.manual_seed(0)
    return Net(3, 64, 256, swap=swap)


def main() -> None:
    base = _build(swap=False)
    head = _build(swap=True)
    example = torch.randn(16, 64)
    vary = [torch.randn(rows, 64) for rows in (16, 32, 64, 128)]

    from torch._inductor import config as inductor_config

    # Install the miscompiling custom pass for the duration of the capture, then restore it. It only
    # affects the inductor-compiled path (graph capture routes through aot_autograd, so the captured
    # graph — and thus the diff and fusion sections — stays clean).
    inductor_config.post_grad_custom_post_pass = _buggy_fusion_pass
    try:
        result = cl.capture(
            head,
            example,
            base=base,
            vary_inputs=vary,
            source=Path(__file__),
            check_divergence=True,
            output_dir=HERE / "_capture",
            **_SESSION,
        )
    finally:
        inductor_config.post_grad_custom_post_pass = None

    shutil.move(str(result.artifact_path), str(HERE / "hero_head.cls.json"))
    assert result.base_path is not None
    shutil.move(str(result.base_path), str(HERE / "hero_base.cls.json"))
    shutil.rmtree(HERE / "_capture", ignore_errors=True)
    print("wrote examples/hero_head.cls.json + examples/hero_base.cls.json")


if __name__ == "__main__":
    main()
