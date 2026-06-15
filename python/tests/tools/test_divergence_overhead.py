"""Hook-overhead characterization for Tool 3's divergence localizer.

This measures and *reports* the cost of the localizer's per-submodule forward hooks. It deliberately
asserts no timing bound at all. The measured overhead is small — single-digit percent — and at
realistic model sizes it sits well inside the run-to-run noise of CPU timing on a shared machine:
repeated measurements of the same workload swing by ±10-15% in either direction (you will even see
"negative overhead", which is physically impossible and is pure scheduling jitter). Worse, on a
*saturated* CPU (a busy CI runner) the ratio of two separately-timed measurements can spike
arbitrarily — we have measured 500-700% — because background load ramps up between the base and the
hooked timings, not because the hooks got slower. Even a loose "< 2x" ratio assertion is therefore
flaky, and a CI flake duly retired it.

So the timing here is report-only. What this test actually *guards* is correctness, and that is
noise-immune: the hooks fire and capture activations during the timed region — the very work the
overhead is the cost of. See the pr08 note for the full investigation (including the under-warmed
first measurement that originally misled us).
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


def test_hook_overhead_is_reported_and_hooks_fire() -> None:
    torch.manual_seed(0)
    eager = _small_transformer()
    compiled = _small_transformer()  # a second model; only `eager`'s hooks fire below
    x = torch.randn(8, 64, 256)

    with torch.no_grad():
        base = _best_time(lambda: eager(x))
        # Register the localizer's hooks once (session enter), then time many forwards. Only
        # `eager`'s submodule hooks fire — we never call `compiled` — so this measures the
        # per-forward capture cost, not the one-off hook registration.
        with divergence_session(eager, compiled) as session:
            hooked = _best_time(lambda: eager(x))

    overhead = (hooked - base) / base
    print(
        f"\nlocalizer hook overhead: {overhead * 100:.1f}% "
        f"(base={base * 1e3:.2f}ms, hooked={hooked * 1e3:.2f}ms)"
    )

    # The only assertion is noise-immune: the hooks actually captured activations during the timed
    # region — the work the overhead is the cost *of*. If capture silently broke (hooks stopped
    # firing) the reported number would be meaningless, and this catches that. A timing ratio would
    # not: on a loaded CPU it flakes regardless of the hooks (see the docstring and the pr08 note).
    assert session.eager_activations, "localizer hooks captured no activations during the timed run"
