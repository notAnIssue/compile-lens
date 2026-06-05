"""Tests for the Mode C dynamo.explain adapter (``_dynamo_explain_adapter``).

Drives the adapter against the committed ``dynamo_explain_output.json`` fixture (the
no-break / single-graph serialized case), a synthetic with-breaks dict, and a live-object
shape, plus the collector wiring (including the ``graph_breaks`` round-trip + schema oracle).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from compile_lens._schema import RedactionPolicy, from_json
from compile_lens.collectors._dynamo_explain_adapter import parse_dynamo_explain
from compile_lens.collectors.recompile import RecompileCollector

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "recompile"
_SCHEMA = json.loads((_REPO_ROOT / "schema" / "v0.5.0.json").read_text())
_EXPLAIN = json.loads((_FIXTURES / "dynamo_explain_output.json").read_text())

_SESSION_KW = {
    "session_id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.6.0",
    "redaction_policy": RedactionPolicy.DEFAULT_STRICT,
}


def test_dynamo_explain_basic() -> None:
    # The fixture: one graph, no breaks. -> 1 compiled graph, 0 graph breaks.
    graph_breaks, compiled_graphs = parse_dynamo_explain(_EXPLAIN)
    assert graph_breaks == []
    assert [g.graph_id for g in compiled_graphs] == ["explain_graph_0"]


def test_dynamo_explain_with_graph_breaks() -> None:
    result = {
        "break_reasons": ["call to builtin print", "data-dependent branch"],
        "graph_break_count": 2,
        "graph_count": 3,
        "ops_per_graph": [["add"], ["mul"], ["relu"]],
    }
    graph_breaks, compiled_graphs = parse_dynamo_explain(result)
    assert [(b.break_id, b.reason) for b in graph_breaks] == [
        ("gb_0", "call to builtin print"),
        ("gb_1", "data-dependent branch"),
    ]
    assert len(compiled_graphs) == 3


def test_dynamo_explain_accepts_live_object() -> None:
    # The live ExplainOutput is an object, not a dict; reason elements expose `.reason`.
    result = SimpleNamespace(
        break_reasons=[SimpleNamespace(reason="graph break X")],
        graph_count=2,
        ops_per_graph=[["a"], ["b"]],
    )
    graph_breaks, compiled_graphs = parse_dynamo_explain(result)
    assert [b.reason for b in graph_breaks] == ["graph break X"]
    assert len(compiled_graphs) == 2


def test_from_dynamo_explain_no_recompile_returns_empty_list(tmp_path: Path) -> None:
    # explain is a single-run structural view -> never contributes recompilations.
    c = RecompileCollector(tmp_path / "s.cls.json", **_SESSION_KW)
    c.from_dynamo_explain(_EXPLAIN)
    artifact = from_json(c.finalize().read_text())
    assert artifact.recompilations == []
    assert len(artifact.compiled_graphs) == 1


def test_from_dynamo_explain_graph_breaks_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "s.cls.json"
    c = RecompileCollector(out, **_SESSION_KW)
    c.from_dynamo_explain(
        {"break_reasons": ["reason A", "reason B"], "graph_count": 1, "ops_per_graph": [["x"]]}
    )
    text = c.finalize().read_text()
    artifact = from_json(text)
    assert [b.reason for b in artifact.graph_breaks] == ["reason A", "reason B"]
    # the graph_breaks-bearing artifact still validates against the JSON Schema oracle
    jsonschema.validate(json.loads(text), _SCHEMA)
