"""``RecompileCollector`` — the Tool 1 collector core.

Tool 1 captures recompilation evidence from a ``torch.compile`` run and writes it to a
``.cls.json`` session artifact that the Rust ``cls-analyzer`` later reads. This module is
the **common pipeline**: it owns artifact assembly + serialization and the dispatch over
the three input modes, while each mode's *parser* lands in a later PR:

- Mode A — ``TORCH_LOGS=+recompiles`` raw text (:meth:`from_logs`)
- Mode B — ``tlparse`` structured output (:meth:`from_tlparse`)
- Mode C — ``torch._dynamo.explain`` result (:meth:`from_dynamo_explain`)

Each mode parses its source into ``list[Recompilation]`` + ``list[CompiledGraph]`` and feeds
them through :meth:`add_records`; :meth:`finalize` assembles a single :class:`ClsArtifact`
and writes it. Secret/path scrubbing of the captured command happens at finalize time in a
later PR (discipline D11); this core records the chosen policy and captures the command
verbatim so that PR has a well-defined input to scrub.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

from compile_lens._schema import (
    SCHEMA_VERSION,
    ClsArtifact,
    CompiledGraph,
    GraphBreak,
    Recompilation,
    RedactionPolicy,
    Session,
    to_json,
)

_log = structlog.get_logger("compile_lens.collectors.recompile")


class Mode(StrEnum):
    """Collection input modes, mirroring the ``cl collect --from-{logs,tlparse,dynamo-explain}``
    mutually-exclusive flag group. Constructing ``Mode(value)`` with an unrecognized value
    raises ``ValueError`` (fail-closed), which is how :meth:`RecompileCollector.collect`
    rejects an unknown mode."""

    LOGS = "logs"
    TLPARSE = "tlparse"
    DYNAMO_EXPLAIN = "dynamo-explain"


def _detect_torch_version() -> str:
    """Best-effort torch version for session metadata. torch is not a hard dependency
    (the user brings their own), so a missing import is normal and yields ``"unknown"``
    rather than raising — a metadata gap must not fail collection."""
    try:
        import torch  # noqa: PLC0415 — optional, intentionally imported lazily

        return str(torch.__version__)
    except Exception:  # pragma: no cover — environment-dependent
        return "unknown"


class RecompileCollector:
    """Accumulates recompile evidence and writes a schema-valid ``.cls.json``.

    Args:
        output_path: where :meth:`finalize` writes the ``.cls.json`` artifact.
        redaction_policy: classification for the captured session (D11). Coerced through
            :class:`RedactionPolicy` so an unknown value fails closed with ``ValueError``
            instead of silently downgrading to a more permissive policy.
        session_id / timestamp / torch_version: session metadata. Each defaults to a
            sensible runtime value when ``None`` (a fresh UUID4, the current UTC time, the
            installed torch version); tests pass them explicitly for deterministic output.
        command: the invoking command line, captured verbatim. A later PR scrubs it
            at finalize.
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
        session_id: str | None = None,
        timestamp: str | None = None,
        torch_version: str | None = None,
        command: str | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.redaction_policy = RedactionPolicy(redaction_policy)  # fail-closed (D11)
        self._session_id = session_id
        self._timestamp = timestamp
        self._torch_version = torch_version
        self._command = command
        self._recompilations: list[Recompilation] = []
        self._compiled_graphs: list[CompiledGraph] = []
        self._graph_breaks: list[GraphBreak] = []

    # ── common pipeline ──────────────────────────────────────────────────────────────
    def add_records(
        self,
        *,
        recompilations: Iterable[Recompilation] = (),
        compiled_graphs: Iterable[CompiledGraph] = (),
        graph_breaks: Iterable[GraphBreak] = (),
    ) -> None:
        """Append parsed records. Additive across calls so each mode contributes its own
        slice without clobbering an earlier one."""
        self._recompilations.extend(recompilations)
        self._compiled_graphs.extend(compiled_graphs)
        self._graph_breaks.extend(graph_breaks)

    # ── modes ────────────────────────────────────────────────────────────────────────
    def from_logs(self, log_path: Path | str) -> None:
        """Mode A — parse a ``TORCH_LOGS=recompiles`` text dump and ingest its recompiles."""
        from compile_lens.collectors._logs_parser import parse_recompiles_log

        text = Path(log_path).read_text()
        self.add_records(recompilations=parse_recompiles_log(text))

    def from_tlparse(self, tlparse_dir: Path | str) -> None:
        """Mode B — adapt a ``tlparse`` output directory (its structured ``raw.jsonl`` +
        ``compile_directory.json``) into recompiles + compiled graphs."""
        from compile_lens.collectors.tlparse_adapter import parse_tlparse_dir

        recompilations, compiled_graphs = parse_tlparse_dir(Path(tlparse_dir))
        self.add_records(recompilations=recompilations, compiled_graphs=compiled_graphs)

    def from_dynamo_explain(self, explain_result: Any) -> None:
        """Mode C — adapt a ``torch._dynamo.explain`` result (the object or its serialized
        fields) into graph breaks + compiled graphs."""
        from compile_lens.collectors._dynamo_explain_adapter import parse_dynamo_explain

        graph_breaks, compiled_graphs = parse_dynamo_explain(explain_result)
        self.add_records(graph_breaks=graph_breaks, compiled_graphs=compiled_graphs)

    def collect(self, mode: Mode | str, source: Any) -> None:
        """Dispatch to the handler for ``mode``. An unrecognized mode raises ``ValueError``
        (caller error); a recognized-but-unbuilt mode raises ``NotImplementedError`` from
        its handler (a later PR fills it in). The two are deliberately distinct signals."""
        match Mode(mode):
            case Mode.LOGS:
                self.from_logs(source)
            case Mode.TLPARSE:
                self.from_tlparse(source)
            case Mode.DYNAMO_EXPLAIN:
                self.from_dynamo_explain(source)

    # ── serialization ────────────────────────────────────────────────────────────────
    def _build_session(self) -> Session:
        # ``command`` is a plain (non-nullable) schema string: pass it only when present so
        # ``to_json``'s exclude_unset drops the key entirely rather than emitting ``null``.
        optional = {"command": self._command} if self._command is not None else {}
        return Session(
            id=self._session_id or str(uuid.uuid4()),
            timestamp=self._timestamp or datetime.now(UTC).isoformat(),
            torch_version=self._torch_version or _detect_torch_version(),
            redaction_policy=self.redaction_policy,
            **optional,
        )

    def finalize(self) -> Path:
        """Assemble the accumulated records into a :class:`ClsArtifact` and write it to
        :attr:`output_path`. Returns the path written."""
        # graph_breaks defaults to [] in the schema; pass it only when non-empty so the
        # exclude_unset serializer keeps a Mode-A/B artifact byte-identical to before.
        optional: dict[str, list[GraphBreak]] = (
            {"graph_breaks": self._graph_breaks} if self._graph_breaks else {}
        )
        artifact = ClsArtifact(
            schema_version=SCHEMA_VERSION,
            session=self._build_session(),
            recompilations=self._recompilations,
            compiled_graphs=self._compiled_graphs,
            **optional,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(to_json(artifact))
        _log.debug(
            "recompile_collector.finalize",
            output_path=str(self.output_path),
            recompilations=len(self._recompilations),
            compiled_graphs=len(self._compiled_graphs),
            graph_breaks=len(self._graph_breaks),
            redaction_policy=str(self.redaction_policy),
        )
        return self.output_path
