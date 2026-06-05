"""Mode C — adapt a ``torch._dynamo.explain`` result into ``GraphBreak`` + ``CompiledGraph``.

``torch._dynamo.explain(fn)(*args)`` returns an ``ExplainOutput`` describing **one** trace:
how many graphs Dynamo produced, how many graph breaks it hit and why, and the ops per
graph. It is a single-run structural explanation — it does **not** report recompilations
(those only appear across repeated calls over time, which Mode A / Mode B capture), so this
adapter contributes ``graph_breaks`` + ``compiled_graphs`` and never ``recompilations``.

Accepts either the live ``ExplainOutput`` object (attribute access) or its serialized dict
(the shape ``tests/fixtures/recompile/_generate.py`` writes for the fixture). Fields read:
``break_reasons`` (→ one :class:`GraphBreak` each), ``graph_count`` + ``ops_per_graph``
(→ one :class:`CompiledGraph` each). Tolerant of either container shape; missing fields
degrade to empty.
"""

from __future__ import annotations

from typing import Any

from compile_lens._schema import CompiledGraph, GraphBreak


def _field(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from a dict (``.get``) or an object (``getattr``)."""
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _break_reason_text(reason: Any) -> str:
    """A break reason is a plain string (serialized form) or a ``GraphCompileReason``
    object (live form) carrying a ``.reason`` attribute."""
    if isinstance(reason, str):
        return reason
    return str(_field(reason, "reason", reason))


def parse_dynamo_explain(
    explain_result: Any,
) -> tuple[list[GraphBreak], list[CompiledGraph]]:
    """Parse a ``torch._dynamo.explain`` result into ``(graph_breaks, compiled_graphs)``."""
    break_reasons = _field(explain_result, "break_reasons", [])
    graph_breaks = [
        GraphBreak(break_id=f"gb_{i}", reason=_break_reason_text(reason))
        for i, reason in enumerate(break_reasons)
    ]

    ops_per_graph = _field(explain_result, "ops_per_graph", [])
    graph_count = _field(explain_result, "graph_count", len(ops_per_graph))
    compiled_graphs = [CompiledGraph(graph_id=f"explain_graph_{i}") for i in range(graph_count)]

    return graph_breaks, compiled_graphs
