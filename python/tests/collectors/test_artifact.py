"""Tests for compile_lens.collectors.artifact.

Two layers: deterministic unit tests of the FX-node serialization (no torch needed, via a
duck-typed fake node) and integration tests that drive a real ``torch.compile`` capture
end-to-end. The integration tests skip cleanly if torch is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from compile_lens._schema import CompiledGraph, from_json
from compile_lens.collectors.artifact import (
    CompileArtifactCollector,
    _iter_node_refs,
    _op_type,
    _serialize_node,
)


# ── duck-typed FX node for deterministic unit tests ─────────────────────────────────────
class FakeNode:
    """Mimics the slice of ``torch.fx.Node`` the serializer reads: ``name`` / ``op`` /
    ``target`` / ``args`` / ``kwargs``."""

    def __init__(
        self,
        name: str,
        op: str,
        target: Any = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> None:
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}


def test_op_type_maps_targets_and_structural_markers() -> None:
    # A call_function uses its target's string form.
    assert _op_type(FakeNode("n", "call_function", target="aten.sub.Tensor")) == "aten.sub.Tensor"
    # Structural markers use the node kind, not the target.
    assert _op_type(FakeNode("p", "placeholder")) == "placeholder"
    assert _op_type(FakeNode("o", "output")) == "output"


def test_inputs_preserve_operand_order() -> None:
    a = FakeNode("a", "placeholder")
    b = FakeNode("b", "placeholder")
    # sub(a, b): the two operands are distinct positions; order must survive.
    sub = FakeNode("sub", "call_function", target="aten.sub.Tensor", args=(a, b))
    node = _serialize_node(sub, FakeNode)
    assert node.inputs == ["a", "b"]
    # The reversed operand order is a genuinely different node, never normalized to a set.
    assert node.inputs != ["b", "a"]


def test_nested_args_are_flattened_in_order() -> None:
    x = FakeNode("x", "call_function", target="aten.add.Tensor")
    # An output node's args look like ((x,),): a nested tuple the serializer must flatten.
    out = FakeNode("output", "output", args=((x,),))
    assert [r.name for r in _iter_node_refs(out.args, FakeNode)] == ["x"]


def test_scalar_args_and_kwargs_become_attrs() -> None:
    a = FakeNode("a", "placeholder")
    # add(a, b=a, alpha=2): a Node ref is an input; the scalar alpha is an attr.
    node = _serialize_node(
        FakeNode(
            "add", "call_function", target="aten.add.Tensor", args=(a, 5), kwargs={"alpha": 2}
        ),
        FakeNode,
    )
    assert node.inputs == ["a"]  # only the Node ref, not the scalar 5
    assert node.attrs == {"arg1": 5, "alpha": 2}


def test_empty_inputs_and_attrs_are_omitted() -> None:
    # A placeholder has no operands and no scalar args, so neither field is emitted.
    node = _serialize_node(FakeNode("p", "placeholder"), FakeNode)
    dumped = node.model_dump(exclude_unset=True)
    assert "inputs" not in dumped
    assert "attrs" not in dumped


# ── redaction (no torch needed) ─────────────────────────────────────────────────────────
def test_default_strict_policy_recorded(tmp_path: Path) -> None:
    out = tmp_path / "session.cls.json"
    collector = CompileArtifactCollector(
        out,
        session_id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-05-21T10:30:00Z",
        torch_version="2.6.0",
    )
    collector._compiled_graphs.append(CompiledGraph(graph_id="graph_0"))
    collector.finalize()
    artifact = from_json(out.read_text())
    assert artifact.session.redaction_policy.value == "default-strict"


def test_strict_refuses_credential_path(tmp_path: Path) -> None:
    """A captured artifact path inside a credential directory is refused at finalize, not
    written with the secret leaked (D11 / CLS-E0008)."""
    from compile_lens.security.redactor import SensitivePathError

    collector = CompileArtifactCollector(
        tmp_path / "session.cls.json",
        session_id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-05-21T10:30:00Z",
        torch_version="2.6.0",
    )
    collector._compiled_graphs.append(
        CompiledGraph(graph_id="graph_0", fx_graph_path="/home/alice/.ssh/keymodel.py")
    )
    with pytest.raises(SensitivePathError):
        collector.finalize()


# ── integration: real torch.compile capture ─────────────────────────────────────────────
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _reset_dynamo() -> None:
    """Clear dynamo's compile cache so each test's function is recompiled and our backend runs."""
    torch._dynamo.reset()


def _collector(tmp_path: Path) -> CompileArtifactCollector:
    return CompileArtifactCollector(
        tmp_path / "session.cls.json",
        session_id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-05-21T10:30:00Z",
        torch_version="2.6.0",
    )


def test_collect_single_compile(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.relu(a) + b

    collector.capture(f, torch.randn(4), torch.randn(4))
    out = collector.finalize()

    artifact = from_json(out.read_text())
    assert len(artifact.compiled_graphs) >= 1
    assert artifact.compiled_graphs[0].nodes, "captured graph should carry node-level structure"


def test_captured_nodes_use_aten_op_types(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.sub(a, b)

    collector.capture(f, torch.randn(4), torch.randn(4))
    artifact = from_json(collector.finalize().read_text())
    nodes = artifact.compiled_graphs[0].nodes

    op_types = {n.op_type for n in nodes}
    # Canonical aten op (not a dynamo-level Python builtin) plus the structural markers.
    assert any(t.startswith("aten.sub") for t in op_types), op_types
    assert "placeholder" in op_types
    assert "output" in op_types


def test_operand_order_preserved_through_real_capture(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.sub(a, b)  # non-commutative: operand order is load-bearing

    collector.capture(f, torch.randn(4), torch.randn(4))
    nodes = from_json(collector.finalize().read_text()).compiled_graphs[0].nodes

    placeholders = [n.id for n in nodes if n.op_type == "placeholder"]
    sub = next(n for n in nodes if n.op_type.startswith("aten.sub"))
    # The sub node consumes the two placeholders in their declared order.
    assert sub.inputs == placeholders


# ── multi-iteration capture ─────────────────────────────────────────────────────────────
def test_iterations_1_default(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.sub(a, b)

    collector.capture(f, torch.randn(4), torch.randn(4))  # iterations defaults to 1
    iterations = from_json(collector.finalize().read_text()).iterations
    assert len(iterations) == 1
    assert iterations[0].iteration_index == 0


def test_iterations_10_captured(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.sub(a, b)

    collector.capture(f, torch.randn(4), torch.randn(4), iterations=10)
    iterations = from_json(collector.finalize().read_text()).iterations
    assert len(iterations) == 10
    assert [it.iteration_index for it in iterations] == list(range(10))


def test_cache_hit_tracking(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.sub(a, b)

    # Same input every iteration: the first run compiles, the rest are cache hits.
    collector.capture(f, torch.randn(4), torch.randn(4), iterations=3)
    iterations = from_json(collector.finalize().read_text()).iterations

    assert iterations[0].cache_hit is False  # initial compile, not a hit
    assert iterations[0].recompilation_triggered is False  # ...but not a recompile either
    assert all(it.cache_hit is True for it in iterations[1:])
    assert all(it.recompilation_triggered is False for it in iterations[1:])


def test_output_signature_stable_for_same_input(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    def f(a, b):
        return torch.sub(a, b)

    collector.capture(f, torch.randn(4), torch.randn(4), iterations=5)
    iterations = from_json(collector.finalize().read_text()).iterations

    signatures = {it.output_signature for it in iterations}
    assert len(signatures) == 1  # identical output every iteration
    assert None not in signatures  # a real signature was computed


def test_module_attrs_changed_detected(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    class Counter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def forward(self, x):
            self.step += 1  # mutable attribute drift across iterations
            return torch.relu(x)

    collector.capture(Counter(), torch.randn(4), iterations=3)
    iterations = from_json(collector.finalize().read_text()).iterations

    # The first iteration has no previous to diff against; later iterations see `step` change.
    later = [it for it in iterations[1:] if it.internal_state_snapshot is not None]
    assert any(
        "step" in it.internal_state_snapshot.module_attrs_changed for it in later
    ), "module attribute drift should be detected after the first iteration"
