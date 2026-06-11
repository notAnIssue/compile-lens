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

from typing import Any


class DivergenceSession:
    """Context manager that captures per-submodule output activations from an eager and a compiled
    model, for localizing the first divergent layer.

    Usage (design §8.3)::

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


def divergence_session(model_eager: Any, model_compiled: Any) -> DivergenceSession:
    """Open a divergence session over an eager and a compiled model (design §8.3)."""
    return DivergenceSession(model_eager, model_compiled)
