"""Tests for the fusion-toggle causal experiment (Tool 3, ADR-032).

Two layers, tested separately:

1. The **pure attribution algorithm** (`minimize_responsible_passes`) is driven by *simulated*
   oracles — a deterministic stand-in for "recompile with these passes disabled and re-check
   divergence". This verifies the bisection/minimization logic exhaustively without torch and
   without a real fusion bug (which cannot be manufactured reliably in a unit test). The simulated
   culprit set *is* the synthetic fusion-bug case, modeled at the oracle seam.

2. The **torch-backed oracle** (real inductor toggling + recompile) is tested under
   ``importorskip`` further down: that it flips the real ``torch._inductor.config`` flags, restores
   them, and returns a bool on a real ``torch.compile`` run.
"""

from compile_lens.tools.divergence import (
    CausalAttribution,
    DivergenceFindings,
    minimize_responsible_passes,
)

# Three real, default-on inductor passes (names grounded in torch._inductor.config). The pure
# algorithm only ever treats them as opaque labels, so the tests do not need torch.
CANDIDATES = ["epilogue_fusion", "pattern_matcher", "reorder_for_locality"]


def _culprit_oracle(culprits):
    """A fake recompile-and-recheck oracle.

    Models a divergence that persists until *every* culprit pass is in the disabled set — i.e. the
    bug is present whenever any culprit pass is still enabled. ``disabled`` is the frozenset of
    passes turned off for this probe.
    """
    culprits = set(culprits)

    def diverges(disabled):
        return not culprits.issubset(disabled)

    return diverges


def test_single_culprit_is_isolated() -> None:
    # The bug lives in one pass; disabling just that pass removes the divergence.
    responsible, probes = minimize_responsible_passes(
        CANDIDATES, _culprit_oracle({"pattern_matcher"})
    )
    assert responsible == ["pattern_matcher"]
    assert probes >= 1  # at least the "disable everything" baseline probe


def test_multi_culprit_keeps_both() -> None:
    # Divergence only vanishes when BOTH passes are off → both are individually necessary.
    responsible, _ = minimize_responsible_passes(
        CANDIDATES, _culprit_oracle({"epilogue_fusion", "reorder_for_locality"})
    )
    assert responsible == ["epilogue_fusion", "reorder_for_locality"]  # sorted


def test_inconclusive_when_cause_is_outside_candidates() -> None:
    # No subset of the candidates removes the divergence (the real culprit is not togglable here).
    responsible, _ = minimize_responsible_passes(
        CANDIDATES, _culprit_oracle({"some_pass_we_do_not_toggle"})
    )
    assert responsible == []


def test_result_is_one_minimal() -> None:
    # Interaction case: the divergence is present while EITHER of two passes is enabled, so
    # disabling either one alone suffices. The result must be a *minimal* set (size 1), not "both".
    def diverges(disabled):
        return ("epilogue_fusion" not in disabled) and ("pattern_matcher" not in disabled)

    responsible, _ = minimize_responsible_passes(CANDIDATES, diverges)
    assert len(responsible) == 1
    assert responsible[0] in {"epilogue_fusion", "pattern_matcher"}


def test_no_candidates_is_inconclusive() -> None:
    # Empty candidate set: nothing to toggle, so nothing can be attributed.
    responsible, _ = minimize_responsible_passes([], _culprit_oracle({"anything"}))
    assert responsible == []


def test_findings_suggested_cause_is_writable() -> None:
    # The causal experiment fills DivergenceFindings.suggested_cause (None until attributed).
    f = DivergenceFindings(None, None, 0, 1e-3, 1e-5)
    assert f.suggested_cause is None
    f.suggested_cause = "divergence removed when inductor pass(es) disabled: pattern_matcher"
    assert "pattern_matcher" in f.suggested_cause


def test_attribution_dataclass_shape() -> None:
    a = CausalAttribution(
        responsible_passes=["epilogue_fusion"],
        attributed=True,
        summary="…",
        num_probes=4,
    )
    assert a.attributed
    assert a.responsible_passes == ["epilogue_fusion"]


# ── torch-backed oracle: real inductor toggling (gated on a torch install) ──────────────────────

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from compile_lens.tools.divergence import (  # noqa: E402
    attribute_divergence,
    build_inductor_oracle,
)


def _tiny_model() -> "torch.nn.Module":
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )


def test_torch_oracle_toggles_and_restores_config() -> None:
    """The real oracle flips an inductor flag for the probe, then restores it (no global leak)."""
    from torch._inductor import config as inductor_config

    model = _tiny_model()
    x = torch.randn(2, 4)
    before = inductor_config.epilogue_fusion
    oracle = build_inductor_oracle(model, torch.compile, lambda m: m(x), rtol=1e-3, atol=1e-5)

    result = oracle(frozenset({"epilogue_fusion"}))

    assert isinstance(result, bool)
    assert inductor_config.epilogue_fusion == before  # restored exactly, even after recompiling


def test_clean_model_attributes_nothing() -> None:
    """A correct model does not diverge under compile, so there is nothing to attribute — and the
    experiment says so honestly instead of inventing a cause or crashing."""
    model = _tiny_model()
    x = torch.randn(2, 4)

    attribution = attribute_divergence(model, (x,))

    assert attribution.attributed is False
    assert attribution.responsible_passes == []
    assert "no divergence" in attribution.summary.lower()


def test_attributes_a_real_injected_inductor_fusion_bug() -> None:
    """End-to-end on a *real* compiled-vs-eager divergence (the AC4.4 case).

    We inject a synthetic fusion bug through inductor's real ``post_grad_custom_post_pass``
    extension slot: a custom post-grad pass that scales the graph output by 1.5 — a genuine
    ``aten.mul`` rewrite in the post-grad FX graph, so the *compiled* model really computes
    different numbers than eager. The experiment then recompiles with that slot cleared and must
    attribute the divergence to it. This exercises the whole pipeline (real ``torch.compile``, real
    inductor pass, real recompile, real toggle) rather than a simulated oracle.
    """
    from torch._inductor import config as inductor_config

    aten = torch.ops.aten

    def corrupting_pass(graph) -> None:
        # Scale the value feeding the output node by 1.5, inserted as a real aten.mul node.
        output_node = next(node for node in graph.nodes if node.op == "output")
        result = output_node.args[0]
        result = result[0] if isinstance(result, (tuple, list)) else result
        with graph.inserting_before(output_node):
            scaled = graph.call_function(aten.mul.Tensor, (result, 1.5))
        output_node.replace_input_with(result, scaled)
        graph.lint()

    model = _tiny_model().eval()
    x = torch.randn(2, 4)

    inductor_config.post_grad_custom_post_pass = corrupting_pass
    try:
        attribution = attribute_divergence(
            model, (x,), candidate_passes=["post_grad_custom_post_pass"]
        )
    finally:
        inductor_config.post_grad_custom_post_pass = None  # never leak the bug to other tests

    assert attribution.attributed is True
    assert attribution.responsible_passes == ["post_grad_custom_post_pass"]
    assert "post_grad_custom_post_pass" in attribution.summary
