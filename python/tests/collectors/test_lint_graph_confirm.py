"""Tests for the Layer-2 functionalized-graph confirmation (Tool 4, ADR-032).

Layer 1 (AST) guesses ``in_place_op_on_alias`` from syntax and can over-flag (its simplified alias
tracking can't see a reassignment, for instance). Layer 2 reads the *real* semantics: it traces the
function through AOTAutograd functionalization, where a mutation-through-a-view is materialized as a
``copy_`` / ``*_scatter`` op. A candidate the graph does not back is dropped; with no runnable model
to trace, the scan degrades to the Layer-1 candidates (CI-friendly).
"""

import pytest

torch = pytest.importorskip("torch")

from compile_lens.collectors.lint import LintPatternScanner  # noqa: E402
from compile_lens.collectors.lint_graph import (  # noqa: E402
    confirm,
    graph_confirms_input_mutation,
)


def _names(hits) -> set[str]:
    return {h.pattern_name for h in hits}


# ── the graph primitive ─────────────────────────────────────────────────────────────────


def test_graph_detects_a_real_view_mutation() -> None:
    def mutates(x):
        y = x[0]  # a view of x
        y.add_(1.0)  # in-place on the view -> really mutates x
        return x.sum()

    assert graph_confirms_input_mutation(mutates, (torch.randn(3, 4),)) is True


def test_graph_clears_a_copy_then_mutate() -> None:
    def clean(x):
        y = x.clone()  # a copy
        y.add_(1.0)
        return y.sum()

    assert graph_confirms_input_mutation(clean, (torch.randn(3, 4),)) is False


# ── confirm(): Layer 1 + Layer 2 ────────────────────────────────────────────────────────


def test_graph_confirms_a_real_candidate_keeps_it() -> None:
    # `view` is an unambiguous aliasing op (Layer 1 flags it); the in-place write really mutates x.
    src = "y = x.view(6)\ny.add_(1.0)\n"
    ast_hits = LintPatternScanner().scan(src)
    assert "in_place_op_on_alias" in _names(ast_hits)

    def real(x):
        y = x.view(6)
        y.add_(1.0)
        return x.sum()

    confirmed = confirm(ast_hits, real, (torch.randn(2, 3),))
    assert "in_place_op_on_alias" in _names(confirmed)  # backed by the graph -> kept


def test_graph_corrects_an_ast_false_positive() -> None:
    # Layer 1's simplified alias tracking can't see the reassignment to a clone, so it over-flags;
    # the function does not actually mutate its input, so the graph refutes and the hit is dropped.
    src = "y = x.view(-1)\ny = x.clone()\ny[0] = 1.0\n"
    ast_hits = LintPatternScanner().scan(src)
    assert "in_place_op_on_alias" in _names(ast_hits)  # AST over-flags

    def reassigned(x):
        y = x.view(-1)
        y = x.clone()
        y[0] = 1.0
        return y.sum()

    confirmed = confirm(ast_hits, reassigned, (torch.randn(4),))
    assert "in_place_op_on_alias" not in _names(confirmed)  # graph corrected it


def test_no_trace_degrades_to_ast_candidates() -> None:
    src = "y = x.view(6)\ny.add_(1.0)\n"
    ast_hits = LintPatternScanner().scan(src)
    assert "in_place_op_on_alias" in _names(ast_hits)  # a real Layer-1 candidate exists
    # No runnable model/inputs -> Layer 2 can't run -> candidates unchanged.
    assert confirm(ast_hits, None, None) == ast_hits


def test_untraceable_function_degrades_to_ast() -> None:
    src = "y = x.view(6)\ny.add_(1.0)\n"
    ast_hits = LintPatternScanner().scan(src)

    def untraceable(x):
        raise RuntimeError("cannot trace")

    # A function that fails to trace must not crash the scan or drop the AST candidates.
    assert confirm(ast_hits, untraceable, (torch.randn(3, 4),)) == ast_hits
