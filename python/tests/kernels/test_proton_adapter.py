"""Tests for the proton adapter (Tool 5).

These tests encode the spec for ``compile_lens.kernels.proton_adapter``: turn a profiling
result into a schema ``KernelMeasurements``, preferring proton and falling back to self-timed
CUDA events, and omitting the record entirely when neither is available.

The ``PROTON_TRACE`` fixture mirrors the real shape of ``triton.profiler``'s output, verified
against proton 3.6.0: a JSON array of root trees where each node is
``{children, frame:{name,type}, metrics}``. A ``proton.scope("...")`` node carries an *empty*
``metrics`` and the real timing lives on its leaf kernel children, whose metrics carry the
literal key ``"time (ns)"`` (with the space and unit) plus an invocation ``count``. proton's
profile output is *aggregated* — total time and count, no per-call distribution — so the adapter
can only derive a mean from it; median/p99 are reachable only from the self-timed path that
records each invocation individually.
"""

from __future__ import annotations

import json
import math

from compile_lens._schema import KernelMeasurements
from compile_lens.kernels import proton_adapter

# A two-scope proton trace shaped exactly like a real hatchet profile.
PROTON_TRACE = [
    {
        "frame": {"name": "ROOT", "type": "function"},
        "metrics": {},
        "children": [
            {
                "frame": {"name": "matmul_demo", "type": "function"},
                "metrics": {},  # scope node — no timing of its own
                "children": [
                    {
                        "frame": {"name": "sgemm_kernel", "type": "function"},
                        "metrics": {"count": 20, "device_type": "CUDA", "time (ns)": 530240},
                        "children": [],
                    }
                ],
            },
            {
                "frame": {"name": "add_demo", "type": "function"},
                "metrics": {},
                "children": [
                    {
                        "frame": {"name": "elementwise_add", "type": "function"},
                        "metrics": {"count": 20, "device_type": "CUDA", "time (ns)": 40000},
                        "children": [],
                    }
                ],
            },
        ],
    }
]


class TestParseProtonTrace:
    def test_aggregates_leaf_kernel_time_into_mean_us(self) -> None:
        result = proton_adapter.parse_proton_trace(PROTON_TRACE)
        m = result["sgemm_kernel"]
        assert isinstance(m, KernelMeasurements)
        assert m.source == "proton"
        assert m.iterations == 20
        # 530240 ns / 20 calls / 1000 = 26.512 us
        assert math.isclose(m.mean_us, 26.512, rel_tol=1e-9)
        # proton is aggregated — there is no per-call distribution to derive
        assert m.median_us is None
        assert m.p99_us is None

    def test_collects_every_leaf_kernel(self) -> None:
        result = proton_adapter.parse_proton_trace(PROTON_TRACE)
        assert set(result) == {"sgemm_kernel", "elementwise_add"}

    def test_scope_nodes_are_not_measurements(self) -> None:
        # The named scopes carry empty metrics; they must not surface as kernels.
        result = proton_adapter.parse_proton_trace(PROTON_TRACE)
        assert "matmul_demo" not in result
        assert "add_demo" not in result

    def test_aggregates_same_kernel_across_scopes(self) -> None:
        trace = [
            {
                "frame": {"name": "ROOT", "type": "function"},
                "metrics": {},
                "children": [
                    {
                        "frame": {"name": "s1", "type": "function"},
                        "metrics": {},
                        "children": [
                            {
                                "frame": {"name": "k", "type": "function"},
                                "metrics": {"count": 10, "time (ns)": 10000},
                                "children": [],
                            }
                        ],
                    },
                    {
                        "frame": {"name": "s2", "type": "function"},
                        "metrics": {},
                        "children": [
                            {
                                "frame": {"name": "k", "type": "function"},
                                "metrics": {"count": 30, "time (ns)": 50000},
                                "children": [],
                            }
                        ],
                    },
                ],
            }
        ]
        result = proton_adapter.parse_proton_trace(trace)
        # 60000 ns over 40 calls → 1.5 us mean
        assert result["k"].iterations == 40
        assert math.isclose(result["k"].mean_us, 1.5)

    def test_reads_from_path(self, tmp_path) -> None:
        p = tmp_path / "trace.hatchet"
        p.write_text(json.dumps(PROTON_TRACE))
        result = proton_adapter.parse_proton_trace(p)
        assert "sgemm_kernel" in result

    def test_empty_trace_is_empty(self) -> None:
        assert proton_adapter.parse_proton_trace([]) == {}


class TestSummarize:
    def test_computes_mean_median_p99(self) -> None:
        samples = [10.0, 12.0, 11.0, 100.0]  # microseconds, unsorted
        m = proton_adapter._summarize_us(samples, source="self_timed")
        assert m is not None
        assert m.source == "self_timed"
        assert m.iterations == 4
        assert math.isclose(m.mean_us, 33.25)
        # linear-interpolation percentiles over sorted [10, 11, 12, 100]
        assert math.isclose(m.median_us, 11.5)
        assert math.isclose(m.p99_us, 97.36)

    def test_single_sample(self) -> None:
        m = proton_adapter._summarize_us([7.0], source="self_timed")
        assert m is not None
        assert m.mean_us == m.median_us == m.p99_us == 7.0

    def test_empty_returns_none(self) -> None:
        assert proton_adapter._summarize_us([], source="self_timed") is None


class TestMeasureRouting:
    def test_prefers_proton_when_kernel_present(self) -> None:
        m = proton_adapter.measure(proton_trace=PROTON_TRACE, kernel_name="sgemm_kernel")
        assert m is not None
        assert m.source == "proton"

    def test_substring_match_on_mangled_name(self) -> None:
        # Real kernel names are mangled C++; a substring must still select them.
        trace = [
            {
                "frame": {"name": "ROOT", "type": "function"},
                "metrics": {},
                "children": [
                    {
                        "frame": {
                            "name": "_ZN7cutlass7Kernel2I42cutlass_80_simt_sgemm_128x32_8x5_nn_align1EEvNT_6ParamsE",  # noqa: E501
                            "type": "function",
                        },
                        "metrics": {"count": 20, "time (ns)": 530239},
                        "children": [],
                    }
                ],
            }
        ]
        m = proton_adapter.measure(proton_trace=trace, kernel_name="sgemm")
        assert m is not None
        assert m.source == "proton"

    def test_falls_back_to_self_timed_without_trace(self, monkeypatch) -> None:
        monkeypatch.setattr(proton_adapter, "_cuda_available", lambda: True)
        captured: dict[str, object] = {}

        def fake_self_timed(fn, *, iterations, warmup):
            captured["iterations"] = iterations
            return KernelMeasurements(iterations=iterations, mean_us=5.0, source="self_timed")

        monkeypatch.setattr(proton_adapter, "self_timed", fake_self_timed)
        m = proton_adapter.measure(fn=lambda: None, iterations=7, warmup=2)
        assert m is not None
        assert m.source == "self_timed"
        assert captured["iterations"] == 7

    def test_proton_miss_falls_back_to_self_timed(self, monkeypatch) -> None:
        monkeypatch.setattr(proton_adapter, "_cuda_available", lambda: True)
        monkeypatch.setattr(
            proton_adapter,
            "self_timed",
            lambda fn, **k: KernelMeasurements(mean_us=1.0, source="self_timed"),
        )
        m = proton_adapter.measure(
            proton_trace=PROTON_TRACE, kernel_name="nonexistent", fn=lambda: None
        )
        assert m is not None
        assert m.source == "self_timed"

    def test_unavailable_returns_none(self, monkeypatch) -> None:
        # A callable but no CUDA → omit (None), never a fabricated "unavailable" source.
        monkeypatch.setattr(proton_adapter, "_cuda_available", lambda: False)
        assert proton_adapter.measure(fn=lambda: None) is None

    def test_nothing_to_measure_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(proton_adapter, "_cuda_available", lambda: True)
        assert proton_adapter.measure() is None
