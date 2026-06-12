"""Tests for first-divergence localization (Tool 3): walk layers in execution order, find the first
that disagrees beyond tolerance."""

import pytest

torch = pytest.importorskip("torch")

from compile_lens import divergence_session  # noqa: E402
from compile_lens.tools.divergence import DivergenceFindings, localize_divergence  # noqa: E402


def _model() -> "torch.nn.Module":
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )


def test_identical_models_do_not_diverge() -> None:
    a = _model()
    b = _model()
    b.load_state_dict(a.state_dict())  # byte-identical weights
    x = torch.randn(2, 4)
    with divergence_session(a, b) as div:
        a(x)
        b(x)
    findings = div.report()
    assert not findings.diverged
    assert findings.first_divergent_layer is None
    assert findings.num_layers_compared == 3


def test_localizes_the_first_divergent_layer() -> None:
    a = _model()
    b = _model()
    b.load_state_dict(a.state_dict())  # identical, then perturb layer "2" only
    with torch.no_grad():
        b[2].weight.add_(1.0)
    x = torch.randn(2, 4)
    with divergence_session(a, b) as div:
        a(x)
        b(x)
    findings = div.report()
    # Layers "0" and "1" are identical; the perturbed Linear "2" is the first divergence.
    assert findings.first_divergent_layer == "2"
    assert findings.max_abs_diff is not None and findings.max_abs_diff > 0


def test_localizes_a_non_finite_layer() -> None:
    """A compiled-side bug that makes a layer emit non-finite values (inf/NaN) is the headline Tool
    3 case (numerical-divergence localization). Inject it into one side and confirm the localizer
    pins that layer, not a downstream one the non-finite values then poison."""
    a = _model()
    b = _model()
    b.load_state_dict(a.state_dict())  # identical, then make layer "2" emit non-finite output
    with torch.no_grad():
        b[2].weight.fill_(float("inf"))  # finite input -> inf/NaN at layer "2"
    x = torch.randn(2, 4)
    with divergence_session(a, b) as div:
        a(x)
        b(x)
    findings = div.report()
    assert findings.first_divergent_layer == "2"
    assert findings.diverged
    # Sanity: the compiled side really is non-finite there (a NaN/inf case, not a small drift).
    assert not torch.isfinite(div.compiled_activations["2"]).all()


def test_tolerance_default_vs_custom() -> None:
    # A tiny perturbation: within the default tolerance, but a tight tolerance flags it.
    eager = {"layer": torch.zeros(4)}
    compiled = {"layer": torch.full((4,), 1e-4)}
    assert not localize_divergence(eager, compiled, rtol=1e-3, atol=1e-3).diverged
    flagged = localize_divergence(eager, compiled, rtol=0.0, atol=1e-6)
    assert flagged.diverged
    assert flagged.first_divergent_layer == "layer"


def test_shape_mismatch_is_a_divergence() -> None:
    eager = {"layer": torch.zeros(4)}
    compiled = {"layer": torch.zeros(8)}
    findings = localize_divergence(eager, compiled, rtol=1e-3, atol=1e-5)
    assert findings.first_divergent_layer == "layer"
    assert findings.max_abs_diff is None  # undefined for a shape mismatch


def test_non_tensor_layers_are_skipped() -> None:
    eager = {"a": None, "b": torch.zeros(4)}
    compiled = {"a": None, "b": torch.zeros(4)}
    findings = localize_divergence(eager, compiled, rtol=1e-3, atol=1e-5)
    assert not findings.diverged
    assert findings.num_layers_compared == 1  # only "b" was comparable


def test_findings_is_exported_dataclass() -> None:
    f = DivergenceFindings(None, None, 0, 1e-3, 1e-5)
    assert f.suggested_cause is None  # filled by the causal experiment later


def test_localizes_against_a_real_compiled_module() -> None:
    """The localizer must work when ``model_compiled`` is a genuine ``torch.compile`` module — that
    is the tool's whole purpose. A compiled ``OptimizedModule`` wraps the original under
    ``_orig_mod``, so its submodule paths are prefixed (``_orig_mod.2`` vs eager's ``2``); without
    unwrapping, the two sides never align by name and the localizer silently reports *no*
    divergence. Compile one side for real, perturb a layer, and confirm localization to it."""
    eager = _model()
    src = _model()
    src.load_state_dict(eager.state_dict())  # identical, then perturb layer "2"
    with torch.no_grad():
        src[2].weight.add_(1.0)
    compiled = torch.compile(src)
    x = torch.randn(2, 4)
    with divergence_session(eager, compiled) as div:
        eager(x)
        compiled(x)
    findings = div.report()
    # Both sides captured the same paths (the compiled side unwrapped to align with eager).
    assert div.captured_modules() == ["0", "1", "2"]
    assert findings.first_divergent_layer == "2"
