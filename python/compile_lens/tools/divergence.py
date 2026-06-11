"""Tool 3 — eager-vs-compiled divergence localization (the session + hooks).

When a model produces different numbers under ``torch.compile`` than in eager mode, the painful
part is finding *where* — which layer first diverges. ``divergence_session`` hooks every submodule
of both an eager and a compiled model and records each one's output activation, so a later step can
walk the layers and report the first one whose eager and compiled activations disagree.

This module is the plumbing: register forward hooks on both models, capture activations keyed by
the submodule's qualified name (so the two sides align by path), and clean the hooks up on exit.
The first-divergence localization and the minifier hand-off land in later changes.

``torch`` is a soft dependency (the user's ``torch.compile`` workload already has it). It is
imported lazily inside the hook path, so importing this module never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DivergenceFindings:
    """The result of localizing eager-vs-compiled divergence across a session's captured layers."""

    #: Qualified name of the first layer (in eager execution order) whose eager and compiled
    #: activations disagree beyond tolerance, or ``None`` if none diverged.
    first_divergent_layer: str | None
    #: Max absolute element-wise difference at that layer (``None`` for a shape mismatch, or when
    #: nothing diverged).
    max_abs_diff: float | None
    #: How many layers were numerically compared (both sides present and tensor-valued).
    num_layers_compared: int
    #: The tolerance used.
    rtol: float
    atol: float
    #: Attributed cause — filled by the causal experiment in a later change; ``None`` for now.
    suggested_cause: str | None = None

    @property
    def diverged(self) -> bool:
        return self.first_divergent_layer is not None


class DivergenceSession:
    """Context manager that captures per-submodule output activations from an eager and a compiled
    model, for localizing the first divergent layer.

    Usage (design notes)::

        with divergence_session(model_eager, model_compiled) as div:
            model_eager(x)
            model_compiled(x)
        div.captured_modules()  # the submodule paths captured on both sides
    """

    def __init__(self, model_eager: Any, model_compiled: Any) -> None:
        self._eager = model_eager
        self._compiled = model_compiled
        self._handles: list[Any] = []
        # Qualified submodule name -> detached output activation (or None for non-tensor outputs).
        self.eager_activations: dict[str, Any] = {}
        self.compiled_activations: dict[str, Any] = {}

    def __enter__(self) -> DivergenceSession:
        self._register(self._eager, self.eager_activations)
        self._register(self._compiled, self.compiled_activations)
        return self

    def __exit__(self, *exc: Any) -> None:
        # Returning None never suppresses exceptions raised inside the context.
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register(self, model: Any, store: dict[str, Any]) -> None:
        # Hook every named submodule except the root itself (name == ""). Submodule names are the
        # alignment key: the same architecture gives the same qualified names on both sides.
        for name, module in model.named_modules():
            if not name:
                continue
            self._handles.append(module.register_forward_hook(_make_hook(name, store)))

    def captured_modules(self) -> list[str]:
        """Submodule paths captured on *both* sides — the comparable set the localizer walks."""
        return sorted(set(self.eager_activations) & set(self.compiled_activations))

    def report(self, rtol: float = 1e-3, atol: float = 1e-5) -> DivergenceFindings:
        """Localize the first layer where eager and compiled activations diverge (design notes)."""
        return localize_divergence(self.eager_activations, self.compiled_activations, rtol, atol)


def _make_hook(name: str, store: dict[str, Any]) -> Any:
    """A forward hook that records the module's output activation under ``name``."""

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        store[name] = _snapshot(output)

    return hook


def _snapshot(output: Any) -> Any:
    """A detached copy of an output tensor, or of the first tensor in a tuple/list output.

    Non-tensor outputs snapshot as ``None`` (nothing to compare numerically)."""
    import torch  # noqa: PLC0415 — lazy: hooks run only with torch present

    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item.detach()
    return None


def localize_divergence(
    eager_activations: dict[str, Any],
    compiled_activations: dict[str, Any],
    rtol: float,
    atol: float,
) -> DivergenceFindings:
    """Walk the captured layers in **eager execution order** and return the first one whose eager
    and compiled activations disagree beyond ``(rtol, atol)``.

    Execution order matters: divergence propagates downstream, so the *first* mismatched layer is
    the root site, not just any mismatched one. ``eager_activations`` preserves insertion order,
    which is the order the forward hooks fired — i.e. eager execution order.
    """
    import torch  # noqa: PLC0415 — comparison needs torch; activations are torch tensors

    compared = 0
    for name, eager in eager_activations.items():
        compiled = compiled_activations.get(name)
        if eager is None or compiled is None:
            continue  # a non-tensor output on either side — nothing to compare numerically
        compared += 1
        if eager.shape != compiled.shape:
            # A shape mismatch is itself a divergence (and abs-diff is undefined).
            return DivergenceFindings(name, None, compared, rtol, atol)
        if not torch.allclose(eager, compiled, rtol=rtol, atol=atol):
            max_abs_diff = (eager.float() - compiled.float()).abs().max().item()
            return DivergenceFindings(name, max_abs_diff, compared, rtol, atol)

    return DivergenceFindings(None, None, compared, rtol, atol)


def divergence_session(model_eager: Any, model_compiled: Any) -> DivergenceSession:
    """Open a divergence session over an eager and a compiled model (design notes)."""
    return DivergenceSession(model_eager, model_compiled)
