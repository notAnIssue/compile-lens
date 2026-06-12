"""Hook-overhead benchmark for Tool 3's divergence localizer.

This characterizes the cost of the localizer's per-submodule forward hooks. It deliberately does
*not* assert a tight "<5% slowdown" bound. The measured overhead is *small* — single-digit percent,
and at realistic model sizes it sits inside the run-to-run noise of CPU timing on a shared machine:
repeated measurements of the same workload swing by ±10-15% in either direction (you will even see
"negative overhead", which is physically impossible and is pure scheduling jitter). The signal is
smaller than the noise, so any tight timing assertion would be flaky. Instead this reports the
measured number for inspection and asserts only a noise-immune regression guard. See the pr08 note
for the full investigation (including the under-warmed first measurement that misled us).
"""

import time

import pytest

torch = pytest.importorskip("torch")

from compile_lens import divergence_session  # noqa: E402


def _small_transformer() -> "torch.nn.Module":
    layer = torch.nn.TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=512, batch_first=True
    )
    return torch.nn.TransformerEncoder(layer, num_layers=2).eval()


def _best_time(fn, warmup: int = 10, iters: int = 50) -> float:
    """Minimum wall-clock over ``iters`` runs (after warmup) — the least-noisy timing statistic."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_hook_overhead_is_bounded_and_reported() -> None:
    torch.manual_seed(0)
    eager = _small_transformer()
    compiled = _small_transformer()  # a second model; only `eager`'s hooks fire below
    x = torch.randn(8, 64, 256)

    with torch.no_grad():
        base = _best_time(lambda: eager(x))
        # Register the localizer's hooks once (session enter), then time many forwards. Only
        # `eager`'s submodule hooks fire — we never call `compiled` — so this measures the
        # per-forward capture cost, not the one-off hook registration.
        with divergence_session(eager, compiled):
            hooked = _best_time(lambda: eager(x))

    overhead = (hooked - base) / base
    print(
        f"\nlocalizer hook overhead: {overhead * 100:.1f}% "
        f"(base={base * 1e3:.2f}ms, hooked={hooked * 1e3:.2f}ms)"
    )

    # Regression guard only — noise-immune. The true overhead is small (single-digit, swamped by
    # CPU jitter; see pr08), so a tight <5% gate would be flaky. < 2x catches only a catastrophic
    # regression (e.g. an accidental tensor copy per hook) that no amount of noise could fake.
    assert hooked < 2.0 * base, (
        f"hooks more than doubled forward time ({overhead * 100:.0f}% overhead) — a real "
        "regression, not measurement noise"
    )
