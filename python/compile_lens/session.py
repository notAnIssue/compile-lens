"""``cl.session()`` — the hero entry point and the **source of truth** (P7, ADR-026).

Six lines run the collectors over a `torch.compile`, produce a `.cls.json`, and render the report::

    import compile_lens as cl

    with cl.session() as s:
        output = model(input)
    s.report().save_html("report.html")   # the hero HTML report (s.artifact_path = the .cls.json)

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
import subprocess
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

    def report(
        self,
        *,
        base: Path | str | None = None,
        gpu: str | None = None,
        db: Path | str | None = None,
    ) -> Report:
        """Return a :class:`Report` for the captured session — the last link of the hero loop::

            with cl.session() as s:
                output = model(input)
            s.report().save_html("report.html")

        The renderer is the ``cl session report`` CLI, driven over a subprocess (a JSON artifact in,
        HTML out — never an in-process binding, ADR-006). ``base`` adds the IR-diff regression
        section; ``gpu`` and ``db`` feed the roofline and lint sections.
        """
        if self.artifact_path is None:
            raise RuntimeError(
                "report() is only available after the `with` block exits and writes the artifact"
            )
        return Report(
            self.artifact_path,
            base=Path(base) if base is not None else None,
            gpu=gpu,
            db=Path(db) if db is not None else None,
        )


def _resolve_cl_bin() -> str:
    """The `cl` binary to drive: ``$CL_BIN`` if set, else ``cl`` on PATH (the convention the
    autotune harness also uses)."""
    return os.environ.get("CL_BIN", "cl")


class Report:
    """A renderable hero report for a captured session.

    The renderer is the Rust ``cl session report`` CLI; this class drives it over a subprocess — a
    ``.cls.json`` artifact in, HTML out, never an in-process binding (ADR-006). Build one via
    :meth:`Session.report`, then call :meth:`save_html` (write a file) or :meth:`html` (get a str).
    """

    def __init__(
        self,
        artifact_path: Path,
        *,
        base: Path | None = None,
        gpu: str | None = None,
        db: Path | None = None,
    ) -> None:
        self.artifact_path = artifact_path
        self._base = base
        self._gpu = gpu
        self._db = db

    def _command(self, output: Path | None) -> list[str]:
        """The ``cl session report`` argv (``--output`` is added only when writing to a file)."""
        cmd = [_resolve_cl_bin(), "session", "report", str(self.artifact_path)]
        if self._base is not None:
            cmd += ["--base", str(self._base)]
        if self._gpu is not None:
            cmd += ["--gpu", self._gpu]
        if self._db is not None:
            cmd += ["--db", str(self._db)]
        if output is not None:
            cmd += ["--output", str(output)]
        return cmd

    def _run(self, output: Path | None) -> str:
        proc = subprocess.run(self._command(output), capture_output=True, text=True)
        if proc.returncode != 0:
            # Surface the CLI's typed error (e.g. an unknown --gpu) rather than an opaque exit code.
            raise RuntimeError(
                f"`cl session report` failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        return proc.stdout

    def html(self) -> str:
        """Render and return the HTML as a string (nothing written to disk)."""
        return self._run(output=None)

    def save_html(self, output: Path | str) -> Path:
        """Render and write the HTML to ``output``; returns the path written."""
        out = Path(output)
        self._run(out)
        return out


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
