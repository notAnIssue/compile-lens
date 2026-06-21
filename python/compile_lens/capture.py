"""``cl.capture()`` — the active, tool-driven one-call capture.

``cl.session()`` is *passive*: a context manager that tees torch's recompile log around your own
model calls. It can only see what crosses stderr, so it fills exactly one part of the report
(recompilations) and nothing that needs the model itself.

``cl.capture()`` is *active*: you hand it the model and a representative input, and it drives every
built collector in one call, assembling a single ``.cls.json`` the hero report renders end to end:

- **graph capture** (Tool 2a diff / Tool 6 fusion) — the aten FX graph, via the
  ``CompileArtifactCollector``;
- **per-iteration cache behaviour** (Tool 2b) — repeated same-shape runs, so the report can show the
  cache was stable (or, on a real stale-cache bug, that it was not);
- **the recompile log under shape variation** (Tool 1) — the model run across ``vary_inputs`` with
  torch's ``recompiles`` log teed and parsed;
- **a static lint scan** (Tool 4) — the source file(s) scanned for correctness anti-patterns;
- **eager-vs-compiled divergence** (Tool 3) — the first layer whose compiled output disagrees with
  eager, localized via forward hooks.

Roofline (Tool 5) is deliberately *not* here: it needs a measured GPU kernel profile, which a CPU
capture cannot produce. It is collected separately against real hardware.

Why a separate active entry rather than extending ``cl.session()``: the report's sections come from
distinct collectors, and "one capture filled the whole report" is only an honest claim if one call
actually runs them. The passive context-manager form never receives the model, so it structurally
cannot (see ADR-041).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compile_lens._schema import (
    SCHEMA_VERSION,
    ClsArtifact,
    RedactionPolicy,
    Session,
    to_json,
)
from compile_lens.session import Report, _StderrTee

if TYPE_CHECKING:
    from compile_lens._schema import Divergence, Recompilation


def _load(path: Path) -> ClsArtifact:
    """Reload a collector's written artifact (already redaction-normalized at its finalize)."""
    return ClsArtifact.model_validate_json(path.read_text())


@dataclass
class CaptureResult:
    """The outcome of a :func:`capture`: where the head (and optional base) ``.cls.json`` landed,
    plus a :meth:`report` shortcut that pre-wires the base for the IR-diff section."""

    artifact_path: Path
    base_path: Path | None = None

    def report(
        self,
        *,
        base: Path | str | None = None,
        gpu: str | None = None,
        db: Path | str | None = None,
    ) -> Report:
        """A :class:`~compile_lens.session.Report` for the captured session. ``base`` defaults to
        the base artifact captured alongside (if any), so the IR-diff section renders without
        re-passing it."""
        resolved_base = base if base is not None else self.base_path
        return Report(
            self.artifact_path,
            base=Path(resolved_base) if resolved_base is not None else None,
            gpu=gpu,
            db=Path(db) if db is not None else None,
        )


def _capture_recompiles(model: Any, vary_inputs: list[Any]) -> list[Recompilation]:
    """Run ``model`` once per input in ``vary_inputs`` and parse torch's recompile log.

    Recompilation is a **dynamo** concept — a guard failure forces a re-trace regardless of the
    backend — so this compiles with ``backend="eager"``: dynamo still guards and logs each
    recompile, but no inductor runs. That keeps the recompile capture fast and, crucially,
    independent of any inductor pass the caller has installed to study a *different* tool (a
    miscompiling custom pass set up for the divergence check must not perturb the recompile count).
    ``dynamic=False`` keeps each new shape a fresh specialization rather than being collapsed by
    automatic dynamic-shape promotion. The stderr fd is teed (torch's ``[__recompiles]`` marker is
    emitted below the Python logging layer), then handed to the same parser ``cl.session()`` uses.
    """
    import torch  # noqa: PLC0415 — optional dependency, imported lazily

    from compile_lens.collectors._logs_parser import parse_recompiles_log

    torch._dynamo.reset()
    try:
        torch._logging.set_logs(recompiles=True)
    except Exception:  # pragma: no cover — environment-dependent
        pass
    tee = _StderrTee().__enter__()
    try:
        compiled = torch.compile(model, backend="eager", dynamic=False)
        with torch.no_grad():
            for x in vary_inputs:
                compiled(x)
    finally:
        tee.__exit__(None, None, None)
    return list(parse_recompiles_log(tee.text))


def _capture_divergence(
    model: Any, example_input: Any, *, rtol: float, atol: float
) -> list[Divergence]:
    """Localize the first eager-vs-compiled divergent layer; empty when the two agree.

    Eager and compiled must be *distinct* module objects sharing weights, or one model's forward
    hooks would fire for both runs and corrupt the comparison — so the compiled side is a deep copy.
    A clean model returns no divergence (the honest empty case); a real compiled-vs-eager bug (a
    miscompiling pass, a bad custom fusion) surfaces as the first layer that disagrees.
    """
    import torch  # noqa: PLC0415

    from compile_lens.tools.divergence import divergence_session, divergence_to_record

    torch._dynamo.reset()
    eager = model
    compiled = torch.compile(copy.deepcopy(model), dynamic=False)
    with torch.no_grad(), divergence_session(eager, compiled) as ds:
        eager(example_input)
        compiled(example_input)
    findings = ds.report(rtol=rtol, atol=atol)
    if not findings.diverged:
        return []
    return [divergence_to_record(findings, divergence_id="div_001")]


def capture(
    model: Any,
    example_input: Any,
    *,
    base: Any | None = None,
    vary_inputs: list[Any] | None = None,
    source: Path | str | None = None,
    check_divergence: bool = False,
    iterations: int = 3,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    output_dir: Path | str = ".compile-lens/",
    redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
    session_id: str | None = None,
    timestamp: str | None = None,
    torch_version: str | None = None,
) -> CaptureResult:
    """Drive every built collector over ``model`` and assemble one ``.cls.json``.

    Args:
        model: the model to diagnose (the "head").
        example_input: a representative input — used for the graph capture, the per-iteration cache
            probe, and (when enabled) the divergence check.
        base: an optional baseline model; when given, its graph is captured to ``base.cls.json`` so
            the report's IR-diff section can compare base→head.
        vary_inputs: inputs of differing shape to drive a recompile; ``None`` skips Tool 1.
        source: a ``.py`` file or directory to lint-scan; ``None`` skips Tool 4.
        check_divergence: run the eager-vs-compiled localization (Tool 3).
        iterations: how many same-shape runs to record for the cache-stability section.
        rtol / atol: tolerances for the divergence comparison.
        output_dir: where the artifacts are written.
        redaction_policy / session_id / timestamp / torch_version: session controls; the latter
            three default to runtime values and are pinned by callers (the hero) for determinism.

    Returns:
        A :class:`CaptureResult` pointing at the head (and optional base) artifact.
    """
    import torch  # noqa: PLC0415

    from compile_lens.collectors.artifact import CompileArtifactCollector
    from compile_lens.collectors.lint_collect import LintCollector

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tv = torch_version or str(torch.__version__)
    sess_kw: dict[str, Any] = {
        "redaction_policy": redaction_policy,
        "session_id": session_id,
        "timestamp": timestamp,
        "torch_version": tv,
    }

    # ── graphs + iterations (head) ──────────────────────────────────────────────────────────
    torch._dynamo.reset()
    tmp_head = out / "_head_graphs.cls.json"
    ca = CompileArtifactCollector(tmp_head, **sess_kw)
    ca.capture(model, example_input, iterations=iterations)
    ca.finalize()
    head_gi = _load(tmp_head)

    # ── recompile storm (Tool 1) ────────────────────────────────────────────────────────────
    recompilations = _capture_recompiles(model, vary_inputs) if vary_inputs else []
    # The recompile log carries raw source paths; redact them the way the collectors do at their
    # finalize (D11) — this artifact is assembled here, not through a collector's finalize.
    if recompilations:
        from compile_lens.security import redactor  # noqa: PLC0415

        if redactor.is_strict(RedactionPolicy(redaction_policy)):
            repo = Path.cwd().name
            for rec in recompilations:
                loc = rec.source_location
                if loc is not None and loc.file is not None:
                    loc.file = redactor.normalize_path(loc.file, repo=repo)

    # ── lint scan (Tool 4) ──────────────────────────────────────────────────────────────────
    lint_findings = []
    if source is not None:
        tmp_lint = out / "_lint.cls.json"
        lc = LintCollector(tmp_lint, **sess_kw)
        lc.scan_path(source)
        lc.finalize()
        lint_findings = list(_load(tmp_lint).lint_findings)
        tmp_lint.unlink(missing_ok=True)

    # ── divergence (Tool 3) ─────────────────────────────────────────────────────────────────
    divergences = (
        _capture_divergence(model, example_input, rtol=rtol, atol=atol) if check_divergence else []
    )

    # ── base graph (Tool 2a IR diff) ────────────────────────────────────────────────────────
    base_path: Path | None = None
    if base is not None:
        torch._dynamo.reset()
        base_path = out / "base.cls.json"
        cb = CompileArtifactCollector(base_path, **sess_kw)
        cb.capture(base, example_input, iterations=1)
        cb.finalize()

    # ── assemble the merged head ────────────────────────────────────────────────────────────
    # exclude_unset (to_json) means an array passed empty would still serialize; pass only the
    # non-empty ones so a probe that found nothing leaves its section's data absent, not [].
    arrays: dict[str, Any] = {}
    if head_gi.iterations:
        arrays["iterations"] = head_gi.iterations
    if recompilations:
        arrays["recompilations"] = recompilations
    if lint_findings:
        arrays["lint_findings"] = lint_findings
    if divergences:
        arrays["divergences"] = divergences

    session = Session(
        id=session_id or head_gi.session.id,
        timestamp=timestamp or head_gi.session.timestamp,
        torch_version=tv,
        redaction_policy=RedactionPolicy(redaction_policy),
    )
    merged = ClsArtifact(
        schema_version=SCHEMA_VERSION,
        session=session,
        compiled_graphs=head_gi.compiled_graphs,
        **arrays,
    )
    head_path = out / "session.cls.json"
    head_path.write_text(to_json(merged))
    tmp_head.unlink(missing_ok=True)
    return CaptureResult(artifact_path=head_path, base_path=base_path)
