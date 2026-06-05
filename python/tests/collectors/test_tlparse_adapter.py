"""Tests for the Mode B tlparse adapter (``tlparse_adapter``).

Drives the adapter against the committed ``tlparse_output/`` fixture (a real tlparse capture
of a 4-compile / 3-recompile session) plus small synthetic dirs for the missing-file and
malformed-line paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from compile_lens._schema import RedactionPolicy, from_json
from compile_lens.collectors.recompile import RecompileCollector
from compile_lens.collectors.tlparse_adapter import parse_tlparse_dir

_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "recompile"
_TLPARSE = _FIXTURES / "tlparse_output"


def test_tlparse_detects_recompiles() -> None:
    recs, _ = parse_tlparse_dir(_TLPARSE)
    # frame 0 compiled 4 times (0/0 initial + 0/1..0/3) -> 3 recompiles.
    assert [r.recompilation_id for r in recs] == ["0/1", "0/2", "0/3"]
    assert all(r.compiled_function_id == "f" for r in recs)
    assert [r.occurred_at_step for r in recs] == [1, 2, 3]
    assert all(r.trigger_reason == "guard_failure" for r in recs)


def test_tlparse_compiled_graphs_carry_artifacts() -> None:
    _, graphs = parse_tlparse_dir(_TLPARSE)
    assert [g.graph_id for g in graphs] == ["0/0", "0/1", "0/2", "0/3"]
    # every compile contributes a graph with its dynamo + inductor artifact paths
    for g in graphs:
        assert g.compiled_function_id == "f"
        assert g.fx_graph_path is not None
        assert g.inductor_ir_path is not None


def test_tlparse_guard_list_from_guard_added_fast() -> None:
    _, graphs = parse_tlparse_dir(_TLPARSE)
    by_id = {g.graph_id: g for g in graphs}
    # the symbolic shape guards (e.g. "s77 <= 1024") are mapped onto the owning compile
    assert by_id["0/1"].guard_list  # at least one guard expr captured
    assert any("s" in expr for expr in by_id["0/3"].guard_list)


def test_tlparse_recompile_wall_clock_from_metrics() -> None:
    recs, _ = parse_tlparse_dir(_TLPARSE)
    # backend_compile_time_s from compilation_metrics -> wall_clock_ms (seconds * 1000)
    assert all(r.wall_clock_ms is not None and r.wall_clock_ms > 0 for r in recs)


def test_tlparse_missing_directory_returns_empty(tmp_path: Path) -> None:
    # No compile_directory.json / raw.jsonl -> no records, no raise (missing-section path).
    recs, graphs = parse_tlparse_dir(tmp_path)
    assert recs == []
    assert graphs == []


def test_tlparse_malformed_raw_line_skipped(tmp_path: Path) -> None:
    (tmp_path / "compile_directory.json").write_text(
        json.dumps(
            {
                "[-/-]": {"artifacts": []},
                "[0/0]": {"artifacts": [{"name": "dynamo_output_graph_1.txt", "url": "g0"}]},
                "[0/1]": {"artifacts": [{"name": "dynamo_output_graph_2.txt", "url": "g1"}]},
            }
        )
    )
    (tmp_path / "raw.jsonl").write_text(
        '{"frame_id": 0, "frame_compile_id": 1, "guard_added_fast": {"expr": "s0 <= 8"}}\n'
        "this is not json and must be skipped\n"
        '{"compilation_metrics": {"compile_id": "0/1", "co_name": "g"}}\n'
    )
    recs, graphs = parse_tlparse_dir(tmp_path)  # must not raise
    assert [g.graph_id for g in graphs] == ["0/0", "0/1"]
    assert [r.recompilation_id for r in recs] == ["0/1"]
    assert recs[0].compiled_function_id == "g"


def test_tlparse_tolerates_missing_metrics_and_guards(tmp_path: Path) -> None:
    # A directory-only capture (older / partial tlparse) still yields graphs; attribution
    # and guards just degrade to None / empty rather than failing.
    (tmp_path / "compile_directory.json").write_text(
        json.dumps({"[0/0]": {"artifacts": []}, "[0/1]": {"artifacts": []}})
    )
    recs, graphs = parse_tlparse_dir(tmp_path)
    assert [g.graph_id for g in graphs] == ["0/0", "0/1"]
    assert graphs[0].compiled_function_id is None
    assert graphs[0].guard_list == []
    assert [r.recompilation_id for r in recs] == ["0/1"]


def test_from_tlparse_into_valid_cls_json(tmp_path: Path) -> None:
    out = tmp_path / "session.cls.json"
    c = RecompileCollector(
        out,
        session_id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-01-01T00:00:00Z",
        torch_version="2.6.0",
        redaction_policy=RedactionPolicy.DEFAULT_STRICT,
    )
    c.from_tlparse(_TLPARSE)
    artifact = from_json(c.finalize().read_text())
    assert [r.recompilation_id for r in artifact.recompilations] == ["0/1", "0/2", "0/3"]
    assert len(artifact.compiled_graphs) == 4
