"""Test-first spec for the Tool 1 ``RecompileCollector`` core.

These tests ARE the contract. The implementation that has to satisfy them lives in
``compile_lens/collectors/recompile.py``. What this PR pins:

  * collector construction + a fail-closed redaction default (D11),
  * the common ingest pipeline the three modes feed
    (``add_records`` <- ``list[Recompilation]`` + ``list[CompiledGraph]``),
  * ``finalize()`` writing a schema-valid ``.cls.json``,
  * the collect dispatch rejecting an unknown mode.

Explicitly **out of scope** here (later PRs):

  * Mode A / B / C parsers (``from_logs`` / ``from_tlparse`` / ``from_dynamo_explain``).
    This PR only establishes the surface they plug into.
  * Actual secret scrubbing of the captured command / paths. This PR *captures* the
    command verbatim and records the chosen policy; a later PR scrubs at finalize time.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from compile_lens._schema import (
    CompiledGraph,
    Recompilation,
    RedactionPolicy,
    from_json,
)
from compile_lens.collectors.recompile import RecompileCollector

# repo root = parents[3] of python/tests/collectors/test_recompile.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = json.loads((_REPO_ROOT / "schema" / "v0.5.0.json").read_text())

# Deterministic session metadata so finalize() output is byte-stable across machines
# (mirrors the determinism discipline the fixture corpus and round-trip suite rely on).
_SESSION_KW = {
    "session_id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.6.0",
}


def _make(tmp_path: Path, **overrides: object) -> RecompileCollector:
    out = tmp_path / "session.cls.json"
    return RecompileCollector(out, **{**_SESSION_KW, **overrides})  # type: ignore[arg-type]


def test_collector_init(tmp_path: Path) -> None:
    c = _make(tmp_path)
    # Fail-closed default: absent an explicit choice, the strictest policy applies (D11).
    assert c.redaction_policy is RedactionPolicy.DEFAULT_STRICT
    assert c.output_path == tmp_path / "session.cls.json"


def test_finalize_writes_valid_cls_json(tmp_path: Path) -> None:
    c = _make(tmp_path)
    c.add_records(
        recompilations=[
            Recompilation(recompilation_id="r1", trigger_reason="guard_failure"),
        ],
        compiled_graphs=[CompiledGraph(graph_id="g1")],
    )
    written = c.finalize()

    assert written == c.output_path
    assert written.exists()
    text = written.read_text()

    # 1. round-trips through the pydantic binding (same gate test_schema.py uses)
    artifact = from_json(text)
    assert artifact.schema_version == "0.5.0"
    assert [r.recompilation_id for r in artifact.recompilations] == ["r1"]
    assert [g.graph_id for g in artifact.compiled_graphs] == ["g1"]
    assert artifact.session.id == _SESSION_KW["session_id"]
    assert artifact.session.torch_version == "2.6.0"

    # 2. validates against the JSON Schema oracle (the real AC1.1 surface)
    jsonschema.validate(json.loads(text), _SCHEMA)


def test_finalize_empty_session_is_valid(tmp_path: Path) -> None:
    # A collector that ingested nothing still emits a schema-valid artifact with empty
    # record arrays (mirrors the Rust analyzer's empty-session contract from PR #25).
    written = _make(tmp_path).finalize()
    artifact = from_json(written.read_text())
    assert artifact.recompilations == []
    assert artifact.compiled_graphs == []
    jsonschema.validate(json.loads(written.read_text()), _SCHEMA)


def test_redaction_policy_applied_to_command(tmp_path: Path) -> None:
    # This PR records the chosen policy (fail-closed enum) and captures the command
    # verbatim; scrubbing the command's secrets is a later PR. This pins that the policy
    # round-trips and the command is captured so that PR has something to scrub.
    c = _make(
        tmp_path,
        redaction_policy=RedactionPolicy.INTERNAL,
        command="python train.py --batch-size 32",
    )
    artifact = from_json(c.finalize().read_text())
    assert artifact.session.redaction_policy is RedactionPolicy.INTERNAL
    assert artifact.session.command == "python train.py --batch-size 32"


def test_add_records_accumulates_across_calls(tmp_path: Path) -> None:
    # The common pipeline is additive: each mode appends its parsed records, so two
    # ingest calls compose rather than overwrite.
    c = _make(tmp_path)
    c.add_records(recompilations=[Recompilation(recompilation_id="r1")])
    c.add_records(recompilations=[Recompilation(recompilation_id="r2")])
    artifact = from_json(c.finalize().read_text())
    assert [r.recompilation_id for r in artifact.recompilations] == ["r1", "r2"]


def test_unknown_mode_raises(tmp_path: Path) -> None:
    # The collect dispatch rejects a mode it does not recognize (caller error),
    # distinct from a recognized-but-not-yet-built mode (a later PR).
    c = _make(tmp_path)
    with pytest.raises(ValueError):
        c.collect("not-a-real-mode", tmp_path / "whatever.log")
