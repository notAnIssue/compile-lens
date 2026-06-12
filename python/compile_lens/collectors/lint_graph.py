"""Layer-2 functionalized-graph confirmation for Tool 4 (`compile-lint`, ADR-032).

Layer 1 (the AST scanner) guesses ``in_place_op_on_alias`` from syntax, and its simplified alias
tracking can over-flag — it cannot see a reassignment, for instance. Layer 2 reads the *real*
semantics: it traces the function through AOTAutograd functionalization, where a mutation through a
view is materialized as an explicit ``copy_`` / ``*_scatter`` op writing back to the input. A
candidate the graph does not back is an AST false positive and is dropped; with no runnable model to
trace, the scan degrades to the Layer-1 candidates (CI-friendly).

``torch`` is imported lazily, so importing this module — and the no-trace path of :func:`confirm` —
costs nothing on a machine without it.
"""

from __future__ import annotations

from typing import Any

from compile_lens.collectors.lint import LintHit


def graph_confirms_input_mutation(fn: Any, inputs: tuple[Any, ...]) -> bool:
    """Trace ``fn`` through functionalization; report whether the graph materializes a real input
    mutation.

    When a function mutates an **input** through a view, functionalization re-applies that mutation
    at the end as an explicit ``copy_(input_placeholder, new_value)`` epilogue. That ``copy_`` whose
    destination is a graph **placeholder** is the precise signal of a real input mutation — the
    semantics Layer 1 could only guess at. A mutation of an *intermediate* (a clone, say) also
    materializes a ``*_scatter`` op, but never a ``copy_`` back to an input, so keying on the
    placeholder-targeted ``copy_`` avoids confusing the two. ``make_fx`` captures the graph
    symbolically.
    """
    import torch  # noqa: PLC0415 — lazy: Layer 2 needs torch; the no-trace path does not
    from torch.func import functionalize  # noqa: PLC0415
    from torch.fx.experimental.proxy_tensor import make_fx  # noqa: PLC0415

    cloned = tuple(i.clone() if isinstance(i, torch.Tensor) else i for i in inputs)
    gm = make_fx(functionalize(fn, remove="mutations_and_views"))(*cloned)
    return any(_writes_back_to_input(node) for node in gm.graph.nodes)


def _writes_back_to_input(node: Any) -> bool:
    # functionalization's input-mutation epilogue: a ``copy_`` whose destination is a placeholder.
    if node.op != "call_function" or not str(node.target).endswith("copy_.default"):
        return False
    return bool(node.args) and getattr(node.args[0], "op", None) == "placeholder"


def confirm(hits: list[LintHit], fn: Any, inputs: tuple[Any, ...] | None) -> list[LintHit]:
    """Confirm Layer-1 candidates against the functionalized graph (ADR-032).

    With a runnable ``fn`` + ``inputs``, the ``in_place_op_on_alias`` candidates are kept only if
    the graph backs them with a real input mutation; if the graph shows none, they were AST false
    positives and are dropped. With no model to trace (``fn`` / ``inputs`` is ``None``) — or if the
    function cannot be traced — the scan degrades to the Layer-1 candidates unchanged. Patterns
    other than ``in_place_op_on_alias`` are not graph-confirmable here and pass through.

    Note (v0): confirmation is function-granular — the graph says whether *the function* mutates an
    input, not which line did, so a function with both a real and a spurious candidate keeps both.
    """
    if fn is None or inputs is None:
        return hits
    try:
        mutates = graph_confirms_input_mutation(fn, inputs)
    except Exception:
        # Untraceable function: we can neither confirm nor refute, so keep the candidates rather
        # than silently drop a possibly-real finding.
        return hits
    if mutates:
        return hits
    return [h for h in hits if h.pattern_name != "in_place_op_on_alias"]
