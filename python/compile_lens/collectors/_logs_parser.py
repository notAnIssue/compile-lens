"""Mode A — parse the ``TORCH_LOGS=recompiles`` artifact stream into ``Recompilation``s.

This is the raw-text collector path: it consumes exactly the stderr a user sees when they
run with ``TORCH_LOGS=recompiles`` and turns each recompile block into a structured
:class:`Recompilation`. A block looks like (the glog-style prefix varies across torch
versions / distributed ranks and is deliberately ignored — we key off the ``[__recompiles]``
marker and parse what follows it)::

    … [0/1] [__recompiles] Recompiling function f in /path/model.py:42
    … [0/1] [__recompiles]     triggered by the following guard failure(s):
    … [0/1] [__recompiles]     - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8

Field mapping (to the v0.5 schema; see ADR-029 — a dedicated guard ``source_location``
field is deferred to v0.6, so the function source path is parsed but not stored):

- ``recompilation_id``    ← the recompile's compile id, e.g. ``"0/1"``
- ``compiled_function_id``← the recompiled function's name, e.g. ``"f"``
- ``trigger_reason``      ← ``"guard_failure"`` (the v0.5 vocabulary; the specific
  category — size / dtype / stride — is recoverable from the guard expression text)
- ``failed_guard``        ← the *primary* guard failure (see :func:`_pick_primary`)
- ``occurred_at_step``    ← the recompile counter (the ``N`` in compile id ``frame/N``)

A recompile lists its guard failure against *every* prior cache entry; v0.5 stores one
``failed_guard`` per recompile, so :func:`_pick_primary` selects the failure against the
frame's original compile (``frame/0``) — the divergence from the baseline specialization,
which carries the defining category. Malformed lines are skipped with a warning, never
raised, so one corrupt line cannot lose the rest of the session.
"""

from __future__ import annotations

import re

import structlog

from compile_lens._schema import FailedGuard, Recompilation

_log = structlog.get_logger("compile_lens.collectors._logs_parser")

#: The artifact marker every recompiles line carries; everything before it is the
#: version-/rank-specific log prefix we ignore.
_MARKER = "[__recompiles]"

#: ``Recompiling function <name> in <file>:<line>`` — starts a recompile block.
_RECOMPILING_RE = re.compile(r"Recompiling function (\S+) in (.+):(\d+)\s*$")
#: ``[frame/counter] [__recompiles]`` — the recompile's own compile id.
_COMPILE_ID_RE = re.compile(r"\[(\d+/\d+)\]\s*\[__recompiles\]")
#: ``- <prior_id>: <guard text>`` — one failed guard against a prior cache entry.
_GUARD_FAILURE_RE = re.compile(r"^-\s*(\d+/\d+):\s*(.+?)\s*$")
#: ``<expression>. expected <prev>, actual <new>`` inside a guard text.
_EXPECTED_ACTUAL_RE = re.compile(r"^(.*?)\.\s*expected\s+(.+?),\s*actual\s+(.+?)\.?\s*$")


def _split_expected_actual(guard_text: str) -> tuple[str, str | None, str | None]:
    """Split a guard text into ``(expression, previous_value, new_value)``.

    ``"tensor 'x' size mismatch at index 0. expected 4, actual 8"`` →
    ``("tensor 'x' size mismatch at index 0", "4", "8")``. A guard with no
    ``expected/actual`` clause returns the whole text and ``None`` values.
    """
    m = _EXPECTED_ACTUAL_RE.match(guard_text)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return guard_text.strip(), None, None


def _pick_primary(
    failures: list[tuple[str, str, str | None, str | None]],
) -> tuple[str, str, str | None, str | None] | None:
    """Choose the representative guard failure for a recompile.

    Prefer the failure against the frame's original compile (prior id ``frame/0``) —
    the divergence from the baseline specialization — falling back to the last-listed
    failure. ``None`` when the block listed no failures.
    """
    if not failures:
        return None
    against_base = [f for f in failures if f[0].rsplit("/", 1)[-1] == "0"]
    return against_base[0] if against_base else failures[-1]


def parse_recompiles_log(text: str) -> list[Recompilation]:
    """Parse a ``TORCH_LOGS=recompiles`` stream into one :class:`Recompilation` per block.

    Non-artifact lines (no ``[__recompiles]`` marker) and malformed artifact lines are
    skipped — the parser never raises on bad input.
    """
    recompilations: list[Recompilation] = []
    # Accumulator for the block currently being read: (compile_id, function, failures).
    compile_id: str | None = None
    function: str | None = None
    failures: list[tuple[str, str, str | None, str | None]] = []

    def _flush() -> None:
        if compile_id is None:
            return
        primary = _pick_primary(failures)
        failed_guard = None
        if primary is not None:
            prior_id, expression, previous, new = primary
            failed_guard = FailedGuard(
                guard_id=prior_id,
                expression=expression,
                previous_value=previous,
                new_value=new,
            )
        counter = compile_id.rsplit("/", 1)[-1]
        recompilations.append(
            Recompilation(
                recompilation_id=compile_id,
                compiled_function_id=function,
                trigger_reason="guard_failure",
                failed_guard=failed_guard,
                occurred_at_step=int(counter) if counter.isdigit() else None,
            )
        )

    for line in text.splitlines():
        marker = line.find(_MARKER)
        if marker == -1:
            continue
        body = line[marker + len(_MARKER) :].strip()

        recompiling = _RECOMPILING_RE.search(body)
        if recompiling is not None:
            _flush()
            cid = _COMPILE_ID_RE.search(line)
            compile_id = cid.group(1) if cid is not None else None
            function = recompiling.group(1)
            failures = []
            continue

        if body.startswith("triggered by"):
            continue

        guard = _GUARD_FAILURE_RE.match(body)
        if guard is not None:
            if compile_id is None:
                _log.warning("recompiles_guard_before_header", line=body)
                continue
            expression, previous, new = _split_expected_actual(guard.group(2))
            failures.append((guard.group(1), expression, previous, new))
            continue

        _log.warning("recompiles_line_unrecognized", line=body)

    _flush()
    return recompilations
