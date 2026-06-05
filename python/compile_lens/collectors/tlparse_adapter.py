"""Mode B — adapt a ``tlparse`` output directory into ``Recompilation`` + ``CompiledGraph``.

``tlparse`` parses a ``TORCH_TRACE`` capture into a directory of structured artifacts. We
**wrap** it (P8 wrap-don't-build): we do not re-parse the raw trace, we read what tlparse
already structured:

- ``compile_directory.json`` — the per-compile-id index. Keys are ``"[frame/counter]"``
  (e.g. ``"[0/0]"``, ``"[0/1]"``); each maps to the artifact files tlparse emitted for that
  compile (``dynamo_output_graph_*.txt``, ``inductor_post_grad_graph_*.txt``, …) with their
  relative ``url``s. This is the authoritative compile-id list + graph-artifact source.
- ``raw.jsonl`` — line-delimited structured events. We read ``compilation_metrics`` (function
  attribution + backend compile time) and ``guard_added_fast`` (symbolic guard expressions),
  joined to a compile-id.

Mapping (per compile id ``F/N``):

- one :class:`CompiledGraph` per compile id — ``fx_graph_path`` / ``inductor_ir_path`` from the
  directory artifacts, ``guard_list`` from the ``guard_added_fast`` expressions.
- one :class:`Recompilation` for every compile with ``N >= 1`` (``N == 0`` is the initial
  compile, not a recompile) — attribution + ``wall_clock_ms`` from ``compilation_metrics``.

tlparse is the structured-artifacts source; the human-readable "which guard failed" reason is
Mode A's strength. Mode B therefore leaves ``failed_guard`` unset and contributes the richer
graph/guard structure instead. Tolerant of missing fields across tlparse versions (tested
against tlparse 0.4.x); a malformed ``raw.jsonl`` line is skipped, never raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from compile_lens._schema import CompiledGraph, Recompilation

_log = structlog.get_logger("compile_lens.collectors.tlparse_adapter")


def _compile_id_sort_key(compile_id: str) -> tuple[int, int]:
    frame, _, counter = compile_id.partition("/")
    return (int(frame) if frame.isdigit() else 0, int(counter) if counter.isdigit() else 0)


def _artifact_paths(artifacts: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Pick (fx_graph_path, inductor_ir_path) from a compile's artifact list."""
    fx = inductor = None
    for art in artifacts:
        name = art.get("name", "")
        url = art.get("url")
        if fx is None and name.startswith("dynamo_output_graph"):
            fx = url
        elif inductor is None and name.startswith("inductor_post_grad_graph"):
            inductor = url
    return fx, inductor


def parse_tlparse_dir(
    tlparse_dir: Path | str,
) -> tuple[list[Recompilation], list[CompiledGraph]]:
    """Parse a tlparse output directory into ``(recompilations, compiled_graphs)``."""
    tlparse_dir = Path(tlparse_dir)
    directory = _load_compile_directory(tlparse_dir / "compile_directory.json")
    metrics, guards = _scan_raw_events(tlparse_dir / "raw.jsonl")

    recompilations: list[Recompilation] = []
    compiled_graphs: list[CompiledGraph] = []

    for compile_id in sorted(directory, key=_compile_id_sort_key):
        artifacts = directory[compile_id].get("artifacts", [])
        fx_path, inductor_path = _artifact_paths(artifacts)
        m = metrics.get(compile_id, {})
        function = m.get("co_name")

        compiled_graphs.append(
            CompiledGraph(
                graph_id=compile_id,
                compiled_function_id=function,
                fx_graph_path=fx_path,
                inductor_ir_path=inductor_path,
                guard_list=guards.get(compile_id, []),
            )
        )

        counter = compile_id.rsplit("/", 1)[-1]
        if counter.isdigit() and int(counter) >= 1:
            backend_s = m.get("backend_compile_time_s")
            recompilations.append(
                Recompilation(
                    recompilation_id=compile_id,
                    compiled_function_id=function,
                    trigger_reason="guard_failure",
                    occurred_at_step=int(counter),
                    wall_clock_ms=backend_s * 1000.0 if backend_s is not None else None,
                )
            )

    return recompilations, compiled_graphs


def _load_compile_directory(path: Path) -> dict[str, dict[str, Any]]:
    """Read ``compile_directory.json`` → {compile_id: entry}, dropping the non-frame
    ``[-/-]`` bucket and normalizing ``"[0/0]"`` keys to ``"0/0"``."""
    if not path.exists():
        _log.warning("tlparse_compile_directory_missing", path=str(path))
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        compile_id = key.strip("[]")
        if "/" not in compile_id or compile_id.startswith("-"):
            continue  # the [-/-] global bucket (torch_version etc.), not a compile
        out[compile_id] = entry
    return out


def _scan_raw_events(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Walk ``raw.jsonl`` once → (metrics_by_compile_id, guard_exprs_by_compile_id)."""
    metrics: dict[str, dict[str, Any]] = {}
    guards: dict[str, list[str]] = {}
    if not path.exists():
        _log.warning("tlparse_raw_jsonl_missing", path=str(path))
        return metrics, guards

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            _log.warning("tlparse_raw_line_unparseable", line=line[:120])
            continue

        compile_id = _event_compile_id(event)

        cm = event.get("compilation_metrics")
        if isinstance(cm, dict):
            cm_id = cm.get("compile_id") or compile_id
            if cm_id is not None:
                metrics[cm_id] = cm

        gf = event.get("guard_added_fast")
        if isinstance(gf, dict) and compile_id is not None:
            expr = gf.get("expr")
            if expr:
                guards.setdefault(compile_id, []).append(expr)

    return metrics, guards


def _event_compile_id(event: dict[str, Any]) -> str | None:
    """Build ``"frame/counter"`` from a raw event's ``frame_id`` + ``frame_compile_id``."""
    frame = event.get("frame_id")
    counter = event.get("frame_compile_id")
    if frame is None or counter is None:
        return None
    return f"{frame}/{counter}"
