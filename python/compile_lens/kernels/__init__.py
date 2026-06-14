"""Tool 5 (kernel roofline triage) — the Python side.

The three-layer cost model lives in the Rust ``cls-roofline`` crate; this package is the Python
surface that drives it: the proton / self-timed measurement adapter (:mod:`proton_adapter`) and the
calibrated-pruning autotune harness (:class:`AutotuneHarness`). The harness reaches the model out of
process via ``cl kernel-roofline`` (ADR-006), so importing this package pulls in no GPU dependency.
"""

from compile_lens.kernels.autotune_harness import AutotuneHarness, SweepReport, SweepRow

__all__ = ["AutotuneHarness", "SweepReport", "SweepRow"]
