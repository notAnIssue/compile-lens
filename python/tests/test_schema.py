"""Tests for compile_lens._schema.

Loads the SAME canonical artifacts shipped in schema/examples/ that the Rust crate's
tests use, so the two language bindings can't silently drift from the JSON Schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from compile_lens._schema import (
    RedactionPolicy,
    from_json,
    to_json,
)

# repo root = parents[2] of python/tests/test_schema.py
_EXAMPLES = Path(__file__).resolve().parents[2] / "schema" / "examples"
MINIMAL = (_EXAMPLES / "minimal.cls.json").read_text()
FULL = (_EXAMPLES / "full.cls.json").read_text()


def test_minimal_example_loads() -> None:
    artifact = from_json(MINIMAL)
    assert artifact.schema_version == "0.5.0"
    assert artifact.session.id == "00000000-0000-4000-8000-000000000000"
    assert artifact.session.torch_version == "2.6.0"
    assert artifact.session.redaction_policy is RedactionPolicy.DEFAULT_STRICT
    # Absent optionals are None / empty; record arrays default to [] not null.
    assert artifact.session.compile_config is None
    assert artifact.session.env_snapshot is None
    assert artifact.session.triton_version is None
    assert artifact.kernels == []
    assert artifact.graph_breaks == []


def test_full_example_loads() -> None:
    artifact = from_json(FULL)
    s = artifact.session
    assert s.gpu_name == "NVIDIA H100 80GB HBM3"
    assert s.duration_ms == 45230
    assert s.world_size == 1
    assert s.collector_versions["recompile"] == "0.5.0"

    assert s.compile_config is not None
    assert s.compile_config.fullgraph is False
    assert s.compile_config.dynamic is None  # present-null union field
    assert s.compile_config.backend == "inductor"

    assert s.env_snapshot is not None
    assert s.env_snapshot.random_seed == 42
    assert s.env_snapshot.relevant_env_vars["TORCH_LOGS"] == "+recompiles"

    # one record from each array parsed
    assert len(artifact.graph_breaks) == 1
    assert len(artifact.recompilations) == 1
    assert artifact.compiled_graphs[0].kernel_ids == ["k_001"]
    assert artifact.compile_phases[0].duration_ms == 1200.5
    assert artifact.iterations[0].cache_hit is True
    assert artifact.lint_findings[0].pattern_category == "in_place_op_on_alias"
    assert artifact.roofline_predictions[0].bound_type == "memory"

    # schema `number` -> float even though written as an integer literal
    flops = artifact.kernels[0].features.flops  # type: ignore[union-attr]
    assert isinstance(flops, float)
    assert flops == 1.2e10
    assert artifact.kernels[0].features.num_regs == 168  # type: ignore[union-attr]


def test_fx_nodes_parse_with_ordered_inputs() -> None:
    artifact = from_json(FULL)
    nodes = artifact.compiled_graphs[0].nodes
    assert len(nodes) == 4

    # required-only node: no inputs, no attrs
    assert nodes[0].id == "n0"
    assert nodes[0].op_type == "placeholder"
    assert nodes[0].inputs == []
    assert nodes[0].attrs == {}

    # the sub node keeps inputs in load-bearing order (n0 before n1) plus an attr
    sub = nodes[2]
    assert sub.op_type == "aten.sub"
    assert sub.inputs == ["n0", "n1"]
    assert sub.attrs == {"alpha": 1}
    # order is a sequence, not a set: reversing it is a different node (what the diff relies on)
    assert sub.inputs != ["n1", "n0"]


def test_fx_nodes_absent_omitted_on_dump() -> None:
    # A graph with no captured nodes does not emit a spurious "nodes":[] — exclude_unset
    # mirrors serde's skip_serializing_if so the two languages stay byte-identical.
    minimal_dump = to_json(from_json(MINIMAL))
    assert "nodes" not in minimal_dump


def test_round_trip_python() -> None:
    for text in (MINIMAL, FULL):
        obj1 = from_json(text)
        dumped = to_json(obj1)
        obj2 = from_json(dumped)
        assert obj1 == obj2
        # re-dump is byte-identical (the determinism the round-trip suite relies on)
        assert dumped == to_json(obj2)

    # present-null union fields survive the round trip; absent ones stay absent
    full_dump = to_json(from_json(FULL))
    assert '"dynamic":null' in full_dump
    assert '"random_seed":42' in full_dump
    minimal_dump = to_json(from_json(MINIMAL))
    assert "dynamic" not in minimal_dump
    assert "compile_config" not in minimal_dump


def test_redaction_policy_enum() -> None:
    # each kebab wire value maps to a member
    assert RedactionPolicy("default-strict") is RedactionPolicy.DEFAULT_STRICT
    assert RedactionPolicy("public-safe") is RedactionPolicy.PUBLIC_SAFE
    # serializes back to the kebab string
    art = from_json(MINIMAL)
    assert '"redaction_policy":"default-strict"' in to_json(art)
    # fail-closed: an unknown policy is a hard error, never a silent default
    bad = dict(json.loads(MINIMAL))
    bad["session"]["redaction_policy"] = "wide-open"
    with pytest.raises(ValidationError):
        from_json(json.dumps(bad))


def test_validation_error_helpful() -> None:
    # missing required field names the field
    missing = dict(json.loads(MINIMAL))
    del missing["session"]["id"]
    with pytest.raises(ValidationError) as exc:
        from_json(json.dumps(missing))
    assert "id" in str(exc.value)

    # wrong type is rejected (no silent coercion of a non-numeric string)
    wrong = dict(json.loads(FULL))
    wrong["iterations"][0]["iteration_index"] = "zero"
    with pytest.raises(ValidationError):
        from_json(json.dumps(wrong))


def test_unknown_field_handled() -> None:
    """DR4 parity with the Rust `test_unknown_field_handled`: unknown keys are preserved
    at the two growth points and dropped inside deep records."""
    data = dict(json.loads(FULL))
    data["future_top_level_array"] = [{"x": 1}]
    data["session"]["gpu_arch"] = "hopper"
    data["kernels"][0]["mystery"] = 1  # deep record -> extra="ignore"

    artifact = from_json(json.dumps(data))  # must not raise
    dumped = to_json(artifact)

    # growth points preserve unknown keys through the round trip
    assert '"future_top_level_array"' in dumped
    assert '"gpu_arch"' in dumped
    # deep record accepts but does not preserve the unknown key
    assert '"mystery"' not in dumped
