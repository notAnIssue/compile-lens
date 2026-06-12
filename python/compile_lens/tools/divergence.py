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

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compile_lens import _schema


@dataclass
class MinifierStatus:
    """Whether dynamo's accuracy minifier was engaged for a session."""

    #: True when the dynamo repro config was found and configured.
    available: bool
    #: Why it was not available, if so (e.g. an older torch without the repro config).
    reason: str | None = None


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
        #
        # Unwrap a torch.compile OptimizedModule first. It wraps the real model under `_orig_mod`,
        # which prefixes every submodule path ("_orig_mod.2" instead of "2"). Hooking the wrapper
        # would store paths that never align with the eager side's, so `captured_modules()` would be
        # empty and the localizer would silently report no divergence. The compiled forward runs
        # `_orig_mod`'s submodules, so hooks registered on them fire during the compiled run.
        target = getattr(model, "_orig_mod", model)
        for name, module in target.named_modules():
            if not name:
                continue
            self._handles.append(module.register_forward_hook(_make_hook(name, store)))

    def captured_modules(self) -> list[str]:
        """Submodule paths captured on *both* sides — the comparable set the localizer walks."""
        return sorted(set(self.eager_activations) & set(self.compiled_activations))

    def report(self, rtol: float = 1e-3, atol: float = 1e-5) -> DivergenceFindings:
        """Localize the first layer where eager and compiled activations diverge (design §8.3)."""
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


@contextlib.contextmanager
def accuracy_minifier() -> Iterator[MinifierStatus]:
    """Engage dynamo's **accuracy minifier** for the duration of the block, then restore.

    Once the divergence localizer tells you *which* layer diverges, the next question is a *minimal*
    reproducer. Rather than bisect by hand, set ``torch._dynamo.config.repro_level = 4`` (accuracy
    minification) and ``repro_after = "aot"``: when a compiled model diverges from eager under this
    config, dynamo writes a minimized repro to its repro directory. Compose this around the run::

        with accuracy_minifier() as m, divergence_session(eager, compiled) as div:
            eager(x)
            compiled(x)
        div.report()  # which layer; dynamo writes the minimal repro if it accuracy-failed

    Yields a :class:`MinifierStatus`. If the dynamo repro config is absent (an older torch), it
    no-ops and yields ``available=False`` — localization still works without it. The prior config is
    always restored on exit.
    """
    try:
        from torch._dynamo import config as dynamo_config  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - torch is present under test
        yield MinifierStatus(available=False, reason=f"torch._dynamo unavailable: {exc}")
        return

    if not (hasattr(dynamo_config, "repro_level") and hasattr(dynamo_config, "repro_after")):
        yield MinifierStatus(available=False, reason="dynamo repro config not present")
        return

    prev_level = dynamo_config.repro_level
    prev_after = dynamo_config.repro_after
    dynamo_config.repro_level = 4  # accuracy minification
    dynamo_config.repro_after = "aot"
    try:
        yield MinifierStatus(available=True)
    finally:
        dynamo_config.repro_level = prev_level
        dynamo_config.repro_after = prev_after


def divergence_session(model_eager: Any, model_compiled: Any) -> DivergenceSession:
    """Open a divergence session over an eager and a compiled model (design §8.3)."""
    return DivergenceSession(model_eager, model_compiled)


# ── Fusion-toggle causal experiment (Tool 3, ADR-032) ───────────────────────────────────────────
#
# Localization tells you *where* eager and compiled disagree. The causal experiment goes one step
# further — *which* inductor optimization pass is responsible — by toggling candidate passes off,
# recompiling, and checking whether the divergence disappears. This turns "describe" (the layer)
# into "attribute" (the cause), which is the ADR-032 north star.
#
# Scope (honest, load-bearing): ``torch._inductor.config`` exposes **pass-level** on/off switches,
# not per-node fusion control. So the finest cause we can establish is *which pass(es), when
# disabled, remove the divergence* — never which specific fused node. The tool says "pass-level"
# everywhere and does not pretend otherwise.

#: Default candidate passes to toggle: default-ON inductor optimization passes whose disabling is a
#: meaningful hypothesis for a fusion-introduced divergence. Names are grounded in
#: ``torch._inductor.config`` (torch 2.x); any name absent in the running torch is skipped. The
#: last two are the custom-pass extension slots — where a user's own (or a buggy) post-grad fusion
#: pass lives; they hold a callable rather than a bool, and are disabled by clearing them to None.
DEFAULT_CANDIDATE_PASSES: tuple[str, ...] = (
    "epilogue_fusion",
    "pattern_matcher",
    "reorder_for_locality",
    "split_cat_fx_passes",
    "loop_ordering_after_fusion",
    "post_grad_custom_post_pass",
    "post_grad_custom_pre_pass",
)


@dataclass
class CausalAttribution:
    """The result of the fusion-toggle causal experiment (ADR-032).

    Pass-level attribution only: see the module-level note. ``responsible_passes`` is the minimal
    set of passes whose disabling removes the divergence; an empty set means the experiment could
    not attribute the divergence to any candidate pass (the cause is outside the toggled set, or
    there was no divergence to begin with).
    """

    #: Minimal set of inductor passes whose disabling removes the divergence (sorted, deduplicated).
    responsible_passes: list[str]
    #: True iff disabling ``responsible_passes`` made eager and compiled agree.
    attributed: bool
    #: Human-readable summary, suitable for ``DivergenceFindings.suggested_cause``.
    summary: str
    #: How many recompile-and-recheck probes the experiment ran (cost transparency).
    num_probes: int


def minimize_responsible_passes(
    candidates: Sequence[str],
    diverges_with_disabled: Callable[[frozenset[str]], bool],
) -> tuple[list[str], int]:
    """Find a **minimal** subset of ``candidates`` whose disabling removes the divergence.

    ``diverges_with_disabled(disabled)`` is an oracle: it (re)compiles with exactly the passes in
    ``disabled`` turned off, re-runs eager vs compiled, and returns whether they *still* diverge.
    This function is pure — it knows nothing about torch; all the torch lives in the oracle — so the
    attribution logic is testable with a simulated oracle.

    Algorithm — greedy 1-minimization (a delta-debugging-lite reduction):

    1. Probe ``disabled = all candidates``. If the divergence *persists* even with every candidate
       pass off, none of them is the cause — return ``([], probes)`` (inconclusive).
    2. Otherwise start from the full disabled set and try to re-enable one pass at a time. Re-enable
       a pass (drop it from the disabled set) only if the divergence stays away without it; keep it
       disabled if re-enabling brings the divergence back.

    The result is *1-minimal*: every pass left in the set is individually necessary to keep the
    divergence away. For the common single-culprit case this is exactly the culprit; for an
    interaction it is a minimal set that suffices. Cost is ``1 + len(candidates)`` oracle probes.

    Returns ``(minimal_passes_sorted, num_probes)``.
    """
    probes = 0
    candidate_set = frozenset(candidates)

    # Step 1: if disabling everything does not fix it, the cause is outside our candidate passes.
    probes += 1
    if diverges_with_disabled(candidate_set):
        return [], probes

    # Step 2: greedily re-enable passes that turn out not to be necessary.
    needed = set(candidate_set)
    for pass_name in candidates:
        trial = needed - {pass_name}
        probes += 1
        if diverges_with_disabled(frozenset(trial)):
            # Re-enabling this pass brought the divergence back → it is necessary; keep it disabled.
            continue
        # Divergence stays away without disabling this pass → it is not needed; re-enable it.
        needed = trial

    return sorted(needed), probes


def build_inductor_oracle(
    model_eager: Any,
    compile_fn: Callable[[Any], Any],
    run: Callable[[Any], Any],
    *,
    rtol: float,
    atol: float,
) -> Callable[[frozenset[str]], bool]:
    """Build the real (torch-backed) divergence oracle used by the causal experiment.

    The returned ``oracle(disabled)`` performs one probe: turn the ``disabled`` inductor passes off,
    clear the dynamo compile cache, recompile ``model_eager`` with ``compile_fn``, run both models
    via ``run``, and report whether their outputs still diverge beyond ``(rtol, atol)`` — then
    restore the inductor config exactly. Output-level comparison (not per-layer hooks) is used here
    on purpose: it is robust to how ``torch.compile`` may trace through submodules.

    ``compile_fn(model) -> compiled_model`` is how to compile (typically ``torch.compile``).
    ``run(model) -> output_tensor`` runs a forward pass; the caller supplies it so the oracle never
    has to guess the model's input signature.
    """
    import torch  # noqa: PLC0415 — lazy: torch is this tool's hard dependency but import stays lazy
    from torch._inductor import config as inductor_config  # noqa: PLC0415

    def oracle(disabled: frozenset[str]) -> bool:
        # Only touch flags that exist in the running torch; skip names a newer/older torch lacks.
        present = [name for name in disabled if hasattr(inductor_config, name)]
        saved = {name: getattr(inductor_config, name) for name in present}
        torch._dynamo.reset()  # clear the compile cache so the toggled config takes effect
        try:
            for name in present:
                # Disable a pass: a boolean optimization flag flips to False; a custom-pass slot
                # (which holds a callable) clears to None. inductor treats those slots as active
                # when they are ``is not None``, so False there would be mistaken for a live pass.
                setattr(inductor_config, name, False if isinstance(saved[name], bool) else None)
            compiled = compile_fn(model_eager)
            out_eager = run(model_eager)
            out_compiled = run(compiled)
            if out_eager.shape != out_compiled.shape:
                return True
            return not torch.allclose(out_eager, out_compiled, rtol=rtol, atol=atol)
        finally:
            # Restore every flag to its prior value and clear the cache again, so a probe leaves no
            # global state behind for the next probe (or the rest of the process).
            for name, value in saved.items():
                setattr(inductor_config, name, value)
            torch._dynamo.reset()

    return oracle


def attribute_divergence(
    model_eager: Any,
    inputs: tuple[Any, ...],
    *,
    compile_fn: Callable[[Any], Any] | None = None,
    candidate_passes: Sequence[str] = DEFAULT_CANDIDATE_PASSES,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    findings: DivergenceFindings | None = None,
) -> CausalAttribution:
    """Attribute an eager-vs-compiled divergence to the inductor pass(es) that cause it (ADR-032).

    Run this *after* localization has shown the model diverges under ``torch.compile``. It builds a
    torch-backed oracle, confirms the model diverges with nothing disabled, then minimizes to the
    set of candidate passes whose disabling removes the divergence (pass-level, see module note).

    ``inputs`` is the positional argument tuple the model is called with (``model(*inputs)``).
    ``compile_fn`` defaults to ``torch.compile``. If ``findings`` is given, its ``suggested_cause``
    is stamped with this experiment's summary, completing the Tool 3 output.
    """
    import torch  # noqa: PLC0415 — lazy torch import (this tool's hard dependency)
    from torch._inductor import config as inductor_config  # noqa: PLC0415

    if compile_fn is None:
        compile_fn = torch.compile

    def run(model: Any) -> Any:
        return model(*inputs)

    oracle = build_inductor_oracle(model_eager, compile_fn, run, rtol=rtol, atol=atol)

    # Confirm there is actually a divergence to attribute (nothing disabled). A correct model agrees
    # under compile — say so honestly rather than hunting for a cause that isn't there.
    probes = 1
    if not oracle(frozenset()):
        attribution = CausalAttribution(
            responsible_passes=[],
            attributed=False,
            summary="no divergence observed under torch.compile; nothing to attribute",
            num_probes=probes,
        )
        if findings is not None:
            findings.suggested_cause = attribution.summary
        return attribution

    # Only consider passes that are present AND currently enabled: a boolean flag that is on, or a
    # custom-pass slot holding a callable. An absent flag, an already-off boolean, or an empty
    # (None) custom slot cannot be "disabled further" — toggling it would waste a recompile and
    # could never explain the divergence.
    def _enabled(name: str) -> bool:
        value = getattr(inductor_config, name)
        return value is True or callable(value)

    present = [
        name for name in candidate_passes if hasattr(inductor_config, name) and _enabled(name)
    ]
    responsible, sub_probes = minimize_responsible_passes(present, oracle)
    probes += sub_probes

    if responsible:
        summary = (
            "divergence removed when inductor pass(es) disabled: "
            + ", ".join(responsible)
            + " — pass-level attribution (torch._inductor.config exposes pass-level toggles, "
            "not per-node fusion control)"
        )
        attribution = CausalAttribution(responsible, True, summary, probes)
    else:
        named = ", ".join(present) if present else "none present"
        summary = (
            f"inconclusive: disabling the candidate inductor passes ({named}) did not remove the "
            "divergence; the cause is outside the toggled pass set"
        )
        attribution = CausalAttribution([], False, summary, probes)

    if findings is not None:
        findings.suggested_cause = attribution.summary
    return attribution


def divergence_to_record(
    findings: DivergenceFindings,
    *,
    divergence_id: str,
    attribution: CausalAttribution | None = None,
) -> _schema.Divergence:
    """Map a runtime ``DivergenceFindings`` (+ optional ``CausalAttribution``) into the schema's
    ``Divergence`` record — the persistent form written into a ``.cls.json``'s ``divergences[]``
    (ADR-034). The structured attribution is nested inside the record when present.

    The ``compile_lens._schema`` import is lazy so this module stays cheap to import and free of a
    pydantic import on the hot path; the returned value is a ``_schema.Divergence``.
    """
    from compile_lens import _schema  # noqa: PLC0415 — lazy: keep module import light

    attribution_record = None
    if attribution is not None:
        attribution_record = _schema.DivergenceAttribution(
            attributed=attribution.attributed,
            responsible_passes=list(attribution.responsible_passes),
            summary=attribution.summary,
            num_probes=attribution.num_probes,
        )
    return _schema.Divergence(
        divergence_id=divergence_id,
        first_divergent_layer=findings.first_divergent_layer,
        max_abs_diff=findings.max_abs_diff,
        num_layers_compared=findings.num_layers_compared,
        rtol=findings.rtol,
        atol=findings.atol,
        suggested_cause=findings.suggested_cause,
        attribution=attribution_record,
    )
