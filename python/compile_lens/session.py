"""``cl.session()`` — the hero entry point and the **source of truth** (P7, ADR-026).

Six lines run the collectors over a `torch.compile` and produce a `.cls.json`::

    import compile_lens as cl

    with cl.session() as s:
        output = model(input)
    # s.artifact_path -> the written .cls.json   (s.report() renders it once Tool 7 lands)

`cl.session(probes=...)` is a factory returning a context manager (ADR-026, weighted-matrix
decision (b) "selective probes"): `__enter__` mounts the capture for the selected probes,
`__exit__` parses what was captured and writes the `.cls.json` through the existing
`RecompileCollector` (applying the redaction policy). The CLI's `cl session` is a thin wrapper
over this same API, so the two surfaces cannot drift — **all collection logic lives here**.

**v0.5.0 scope.** Only the `recompile` probe is implemented; it is the default. ADR-026's frozen
default is `{"recompile", "iterations"}`, but `iterations` needs a capture mechanism that does not
yet exist (the session holds no model reference) and lands in its own change — until then it is a
*recognized but unbuilt* probe (a `NotImplementedError`, distinct from an unknown-probe
`ValueError`). The expensive probes (`diff`, `divergence`, `kernels`) are likewise recognized but
unbuilt.

**Why the capture is at the file-descriptor layer.** torch's `recompiles` artifact log carries a
`[__recompiles]` marker added on torch's *own* emit path; a Python `logging.Handler` (even with
torch's formatter) does not reproduce it. The marked text reliably reaches the process **stderr
fd**, so the session tees fd 2 during the `with` and feeds the result to the existing
`parse_recompiles_log` (which is documented to consume exactly that stderr).
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Literal

from compile_lens._schema import RedactionPolicy

#: The probes implemented in v0.5.0. `recompile` is the only working one; it is the default.
_IMPLEMENTED_PROBES = frozenset({"recompile"})
#: Probes that are part of the frozen ADR-026 vocabulary but not yet built. Selecting one is a
#: `NotImplementedError` (a recognized name, a later change), kept distinct from an unknown name.
_UNBUILT_PROBES = frozenset({"iterations", "diff", "divergence", "kernels"})
_KNOWN_PROBES = _IMPLEMENTED_PROBES | _UNBUILT_PROBES

#: ADR-026's default is `{"recompile", "iterations"}`; until `iterations` is built the working
#: default is just `recompile`, so the six-line example runs end to end.
_DEFAULT_PROBES = frozenset({"recompile"})


class _StderrTee:
    """Capture everything written to the stderr **fd** during the block, while passing it through
    to the real stderr (the user's own output is never swallowed).

    fd-level, not `logging`-level: torch's `[__recompiles]` marker is added on torch's own emit
    path and bypasses the Python logging API, but the marked bytes still reach fd 2. A pipe + a
    pump thread tee those bytes to both the real stderr and an in-memory buffer.
    """

    def __init__(self) -> None:
        self._buf = io.BytesIO()
        self._saved_fd = -1
        self._pipe_r = -1
        self._pipe_w = -1
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _StderrTee:
        sys.stderr.flush()
        self._saved_fd = os.dup(2)  # a handle to the real stderr, kept for pass-through + restore
        self._pipe_r, self._pipe_w = os.pipe()
        os.dup2(self._pipe_w, 2)  # everything written to fd 2 now goes to the pipe
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _pump(self) -> None:
        while True:
            chunk = os.read(self._pipe_r, 65536)
            if not chunk:
                break
            os.write(self._saved_fd, chunk)  # pass through to the real stderr
            self._buf.write(chunk)  # and keep a copy

    def __exit__(self, *_exc: object) -> None:
        sys.stderr.flush()
        os.dup2(self._saved_fd, 2)  # restore the real stderr first
        os.close(self._pipe_w)  # EOF for the pump
        if self._thread is not None:
            self._thread.join(timeout=5)
        os.close(self._pipe_r)
        os.close(self._saved_fd)

    @property
    def text(self) -> str:
        return self._buf.getvalue().decode("utf-8", errors="replace")


class Session:
    """The hero session (ADR-026). Construct via :func:`session`, use as a context manager."""

    def __init__(
        self,
        probes: set[str] | frozenset[str] = _DEFAULT_PROBES,
        output_dir: Path | str = ".compile-lens/",
        redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
    ) -> None:
        self.probes = frozenset(probes)
        # Unknown name -> ValueError (caller typo); recognized-but-unbuilt -> NotImplementedError
        # (a later change). The two are deliberately distinct signals (ADR-022).
        unknown = self.probes - _KNOWN_PROBES
        if unknown:
            raise ValueError(
                f"unknown probe(s) {sorted(unknown)}; known probes are {sorted(_KNOWN_PROBES)}"
            )
        unbuilt = self.probes & _UNBUILT_PROBES
        if unbuilt:
            raise NotImplementedError(
                f"probe(s) {sorted(unbuilt)} are not implemented in v0.5.0; "
                f"available: {sorted(_IMPLEMENTED_PROBES)} (see ADR-026)"
            )
        self.output_dir = Path(output_dir)
        self.redaction_policy = RedactionPolicy(redaction_policy)  # fail-closed (D11)
        self.artifact_path: Path | None = None
        self._tee: _StderrTee | None = None

    def __enter__(self) -> Session:
        if "recompile" in self.probes:
            # Best-effort: enable torch's recompiles artifact log. No torch (or an API change) is
            # not fatal — the fd-tee still runs and simply captures no torch markers.
            try:
                import torch  # noqa: PLC0415 — optional, the user brings their own

                torch._logging.set_logs(recompiles=True)
            except Exception:  # pragma: no cover — environment-dependent
                pass
            self._tee = _StderrTee().__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        captured = ""
        if self._tee is not None:
            self._tee.__exit__(exc_type, exc, tb)
            captured = self._tee.text
        self._write_artifact(captured)
        return False  # never suppress an exception raised inside the `with`

    def _write_artifact(self, captured_stderr: str) -> None:
        import socket

        from compile_lens.collectors._logs_parser import parse_recompiles_log
        from compile_lens.collectors.recompile import RecompileCollector

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "session.cls.json"
        collector = RecompileCollector(
            out,
            redaction_policy=self.redaction_policy,
            command=" ".join(sys.argv),
            host=socket.getfqdn(),
        )
        if "recompile" in self.probes and captured_stderr:
            collector.add_records(recompilations=parse_recompiles_log(captured_stderr))
        self.artifact_path = collector.finalize()

    def report(self, output: Path | str | None = None) -> object:
        """Render the session's `.cls.json` into the hero HTML report.

        Not built yet: rendering goes through `cl session report` (Phase 7, a later change). For now
        read the artifact at :attr:`artifact_path`.
        """
        raise NotImplementedError(
            "Session.report() renders via `cl session report`, which is not built yet. "
            "For now, the written artifact is at session.artifact_path."
        )


def session(
    probes: set[str] | frozenset[str] = _DEFAULT_PROBES,
    output_dir: Path | str = ".compile-lens/",
    redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
) -> Session:
    """Open a hero session (ADR-026). A factory returning a context manager::

        with cl.session() as s:
            output = model(input)
        # s.artifact_path -> the written .cls.json

    Args:
        probes: which collectors to mount. Defaults to ``{"recompile"}`` (the only one built in
            v0.5.0). An unknown name raises ``ValueError``; a recognized-but-unbuilt one
            (``iterations`` / ``diff`` / ``divergence`` / ``kernels``) raises
            ``NotImplementedError``.
        output_dir: where the ``.cls.json`` is written.
        redaction_policy: D11 classification applied when the artifact is written (default strict).
    """
    return Session(probes=probes, output_dir=output_dir, redaction_policy=redaction_policy)
