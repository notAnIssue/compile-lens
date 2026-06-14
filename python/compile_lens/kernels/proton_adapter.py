"""``proton_adapter`` — turn a profiling result into a schema ``KernelMeasurements`` (Tool 5).

The roofline cost model calibrates its predicted ranking against *measured* kernel runtimes
(Layer 3). This adapter is where those measurements come from. It has two sources, in the order
the design mandates:

  * **proton** (Triton's official profiler) is primary. After a profiled run, proton writes a
    *hatchet* trace — a JSON array of call trees where each node is
    ``{children, frame:{name,type}, metrics}``. A ``proton.scope(...)`` node is a container with
    empty ``metrics``; the real timing lives on its leaf kernel children, whose metrics carry the
    literal key ``"time (ns)"`` (with the space and unit) and an invocation ``count``. proton
    aggregates, so a leaf gives only total time over ``count`` calls — enough for a **mean**, not
    a per-call distribution. :func:`parse_proton_trace` walks that tree; it is pure JSON handling
    and needs neither proton, torch, nor a GPU installed.
  * **self-timed CUDA events** is the fallback when no proton trace is available.
    :func:`self_timed` records each invocation with a ``torch.cuda.Event`` pair, so it *does*
    yield a full distribution (mean / median / p99). It is the only path that touches torch, and
    torch is imported lazily inside it.

:func:`measure` is the single entry point that encodes the policy: prefer a proton measurement for
the requested kernel; otherwise self-time the supplied callable if CUDA is available; otherwise
return ``None`` so the caller **omits** the ``measurements`` record — there is no fabricated
``"unavailable"`` source, only the schema's ``"proton"`` / ``"self_timed"``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import structlog

from compile_lens._schema import KernelMeasurements

_log = structlog.get_logger("compile_lens.kernels.proton_adapter")

# The literal metric key proton writes on a timed leaf node (space and unit included).
_TIME_KEY = "time (ns)"


# ── proton hatchet trace parsing (pure JSON; no proton/torch/GPU needed) ───────────────────
def parse_proton_trace(
    trace: list[Any] | dict[str, Any] | str | Path,
) -> dict[str, KernelMeasurements]:
    """Walk a proton hatchet trace and return ``{kernel_name: KernelMeasurements}``.

    A node counts as a measured kernel when its ``metrics`` carries ``"time (ns)"`` (scope
    container nodes carry empty metrics and are skipped). Leaves sharing a name — the same kernel
    invoked under several scopes — have their time and count summed, then collapsed into one mean
    (``time_ns / count / 1000``). ``source`` is stamped ``"proton"``; median/p99 stay ``None``
    because proton's output is already aggregated.

    ``trace`` may be the parsed JSON (a list of root trees, or a single root dict) or a path to a
    ``.hatchet`` file.
    """
    if isinstance(trace, (str, Path)):
        trace = json.loads(Path(trace).read_text())

    totals: dict[str, list[float]] = {}  # name -> [time_ns, count]

    def walk(node: Any) -> None:
        metrics = node.get("metrics") or {}
        if _TIME_KEY in metrics:
            name = (node.get("frame") or {}).get("name", "<unknown>")
            entry = totals.setdefault(name, [0.0, 0.0])
            entry[0] += float(metrics[_TIME_KEY])
            entry[1] += float(metrics.get("count", 1))
        for child in node.get("children") or []:
            walk(child)

    roots = trace if isinstance(trace, list) else [trace]
    for root in roots:
        walk(root)

    out: dict[str, KernelMeasurements] = {}
    for name, (time_ns, count) in totals.items():
        if count <= 0:
            continue
        out[name] = KernelMeasurements(
            iterations=int(count),
            mean_us=time_ns / count / 1000.0,
            source="proton",
        )
    return out


# ── statistics shared by the self-timed path ──────────────────────────────────────────────
def _percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted sample (numpy's default method)."""
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = (pct / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summarize_us(samples_us: Iterable[float], *, source: str) -> KernelMeasurements | None:
    """Reduce per-call microsecond samples to a ``KernelMeasurements``; ``None`` if empty."""
    ordered = sorted(samples_us)
    if not ordered:
        return None
    return KernelMeasurements(
        iterations=len(ordered),
        mean_us=sum(ordered) / len(ordered),
        median_us=_percentile(ordered, 50.0),
        p99_us=_percentile(ordered, 99.0),
        source=source,
    )


# ── self-timed CUDA events fallback (lazy torch; runs only when called) ─────────────────────
def _cuda_available() -> bool:
    """True when torch is importable and a CUDA device is present."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover — environment-dependent
        return False


def self_timed(
    fn: Callable[[], Any], *, iterations: int = 50, warmup: int = 10
) -> KernelMeasurements | None:
    """Time ``fn`` with CUDA events: ``warmup`` untimed calls, then ``iterations`` timed ones.

    Each timed call brackets ``fn`` between a recorded start/end event pair and reads the elapsed
    time after a synchronize. ``Event.elapsed_time`` returns **milliseconds**, so it is scaled to
    microseconds. Returns a full distribution (mean / median / p99) tagged ``"self_timed"``.
    """
    import torch  # noqa: PLC0415

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)  # ms → us

    return _summarize_us(samples_us, source="self_timed")


# ── unified policy ─────────────────────────────────────────────────────────────────────────
def _select(parsed: dict[str, KernelMeasurements], kernel_name: str) -> KernelMeasurements | None:
    """Pick a kernel by exact name, else by substring (kernel names are mangled C++)."""
    if kernel_name in parsed:
        return parsed[kernel_name]
    for name, measurement in parsed.items():
        if kernel_name in name:
            return measurement
    return None


def measure(
    *,
    proton_trace: list[Any] | dict[str, Any] | str | Path | None = None,
    kernel_name: str | None = None,
    fn: Callable[[], Any] | None = None,
    iterations: int = 50,
    warmup: int = 10,
) -> KernelMeasurements | None:
    """Measure one kernel, preferring proton and falling back to self-timed CUDA events.

    1. If a ``proton_trace`` is given and ``kernel_name`` resolves in it, return that (proton).
    2. Otherwise, if ``fn`` is given and CUDA is available, self-time it.
    3. Otherwise return ``None`` — the caller omits ``measurements`` rather than inventing a
       source.
    """
    if proton_trace is not None and kernel_name:
        hit = _select(parse_proton_trace(proton_trace), kernel_name)
        if hit is not None:
            return hit
        _log.debug("proton_adapter.proton_miss", kernel_name=kernel_name)

    if fn is not None and _cuda_available():
        return self_timed(fn, iterations=iterations, warmup=warmup)

    _log.debug(
        "proton_adapter.unavailable", has_trace=proton_trace is not None, has_fn=fn is not None
    )
    return None
