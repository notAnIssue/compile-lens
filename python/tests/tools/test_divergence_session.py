"""Tests for the divergence session plumbing (Tool 3): hook register / capture / align / unmount."""

import pytest

torch = pytest.importorskip("torch")

from compile_lens import DivergenceSession, divergence_session  # noqa: E402
from compile_lens.tools.divergence import _snapshot  # noqa: E402


def _model() -> "torch.nn.Module":
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )


def test_factory_returns_session() -> None:
    assert isinstance(divergence_session(_model(), _model()), DivergenceSession)


def test_captures_submodule_activations_aligned_by_path() -> None:
    eager, other = _model(), _model()
    x = torch.randn(2, 4)
    with divergence_session(eager, other) as div:
        eager(x)
        other(x)

    # Every named submodule (but not the root) is captured on both sides, keyed by qualified name.
    assert set(div.eager_activations) == {"0", "1", "2"}
    assert set(div.compiled_activations) == {"0", "1", "2"}
    assert div.captured_modules() == ["0", "1", "2"]
    # Captured activations are detached tensors of the layer's output shape.
    assert div.eager_activations["0"].shape == (2, 8)
    assert not div.eager_activations["0"].requires_grad


def test_hooks_removed_on_exit() -> None:
    eager, other = _model(), _model()
    with divergence_session(eager, other) as div:
        assert div._handles  # hooks registered inside the context
    assert not div._handles  # and removed on exit


def test_does_not_suppress_exceptions() -> None:
    with pytest.raises(ValueError, match="boom"):
        with divergence_session(_model(), _model()):
            raise ValueError("boom")


def test_captured_modules_is_intersection() -> None:
    # Different architectures share only the paths present on both sides.
    a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    b = torch.nn.Sequential(torch.nn.Linear(4, 4))
    x = torch.randn(2, 4)
    with divergence_session(a, b) as div:
        a(x)
        b(x)
    assert div.captured_modules() == ["0"]  # only "0" exists on both


def test_snapshot_handles_tuple_and_non_tensor() -> None:
    t = torch.randn(3)
    snap = _snapshot((t, "metadata"))
    assert torch.equal(snap, t)
    assert _snapshot("not a tensor") is None
