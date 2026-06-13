"""``LintCollector`` — scan Python source for Tool 4 anti-patterns and write a ``.cls.json``.

This is the Python *front-end* half of Tool 4 (``compile-lint``). It drives the Layer-1
:class:`~compile_lens.collectors.lint.LintPatternScanner` over a file or a directory of ``.py``
sources, turns each structural match into a candidate ``lint_findings[]`` entry, and writes a
schema-valid ``.cls.json``. The Rust ``cl compile-lint <artifact>`` then joins those candidates with
the correctness database, assigns severity from the user's torch version, and renders (the design doc
ADR-006: the two halves meet on disk, not over FFI; unifying them into one command is the Phase 7
hero).

**Layer-1 only.** Confirmation against a functionalized FX graph (Layer 2,
:mod:`compile_lens.collectors.lint_graph`) needs the live function plus example inputs to trace,
which a static source scan cannot supply; it stays a programmatic step. So every finding written
here carries ``confidence = "ast_match"`` and a placeholder severity — the analyzer assigns the
real one.

The scanner is pure ``ast`` and never imports torch; building and writing the artifact never import
torch either. Mirrors :class:`compile_lens.collectors.artifact.CompileArtifactCollector`'s session +
redaction handling.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog

from compile_lens._schema import (
    SCHEMA_VERSION,
    ClsArtifact,
    LintFinding,
    RedactionPolicy,
    Session,
    SourceRange,
    to_json,
)
from compile_lens.collectors.lint import LintHit, LintPatternScanner

_log = structlog.get_logger("compile_lens.collectors.lint_collect")


class LintCollector:
    """Scan ``.py`` source for Tool 4 anti-patterns and write the candidates to a ``.cls.json``.

    Args:
        output_path: where :meth:`finalize` writes the ``.cls.json``.
        operator_rules: optional ``operator -> {parameter: pattern_category}`` map forwarded to the
            scanner (ADR-036). Without it only the structural ``in_place_op_on_alias`` fires; the
            operator-family rules come from the correctness database, which the front-end reads and
            passes here (that wiring lands separately — for now this stays an explicit hook).
        redaction_policy: classification for the session (D11), coerced through
            :class:`RedactionPolicy` so an unknown value fails closed.
        session_id / timestamp / torch_version: session metadata; each defaults to a runtime value
            when ``None`` (fresh UUID4, current UTC time, installed torch version), with tests
            passing them explicitly for deterministic output.
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        operator_rules: dict[str, dict[str, str]] | None = None,
        redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
        session_id: str | None = None,
        timestamp: str | None = None,
        torch_version: str | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.redaction_policy = RedactionPolicy(redaction_policy)  # fail-closed (D11)
        self._scanner = LintPatternScanner(operator_rules)
        self._session_id = session_id
        self._timestamp = timestamp
        self._torch_version = torch_version
        self._findings: list[LintFinding] = []

    # ── scan ─────────────────────────────────────────────────────────────────────────────
    def scan_path(self, path: Path | str) -> int:
        """Scan a single ``.py`` file, or every ``.py`` under a directory (recursively, sorted for
        deterministic order). Returns the number of candidate findings this call added."""
        path = Path(path)
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        before = len(self._findings)
        for file in files:
            self._scan_file(file)
        return len(self._findings) - before

    def _scan_file(self, file: Path) -> None:
        for hit in self._scanner.scan(file.read_text(), filename=str(file)):
            self._findings.append(self._to_finding(hit, str(file)))

    def _to_finding(self, hit: LintHit, file: str) -> LintFinding:
        # finding_id is positional in scan order (lf_001, lf_002, …); len() here is the
        # pre-append count, so +1 gives this finding's 1-based index.
        return LintFinding(
            finding_id=f"lf_{len(self._findings) + 1:03d}",
            pattern_category=hit.pattern_name,
            # Layer-1 sees structure only; the Rust analyzer assigns the real severity from the
            # database + the user's torch version, so this is a pre-analysis placeholder.
            severity="candidate",
            source_location=SourceRange(file=file, line_start=hit.line),
            confidence="ast_match",
        )

    # ── serialization ──────────────────────────────────────────────────────────────────────
    def _build_session(self) -> Session:
        def _detect_torch_version() -> str:
            try:
                import torch  # noqa: PLC0415

                return str(torch.__version__)
            except Exception:  # pragma: no cover — environment-dependent
                return "unknown"

        return Session(
            id=self._session_id or str(uuid.uuid4()),
            timestamp=self._timestamp or datetime.now(UTC).isoformat(),
            torch_version=self._torch_version or _detect_torch_version(),
            redaction_policy=self.redaction_policy,
        )

    def finalize(self) -> Path:
        """Assemble the candidate findings into a :class:`ClsArtifact` and write it.

        Under a strict policy each finding's source path is normalized at write time (D11): a
        machine-absolute path collapses to ``{repo}/…`` while a project-relative path — the useful
        part, *where* the anti-pattern is — is kept. Code excerpts are never captured here (opt-in
        only), so no source text leaves the machine."""
        from compile_lens.security import redactor  # noqa: PLC0415

        if redactor.is_strict(self.redaction_policy):
            repo = Path.cwd().name
            for finding in self._findings:
                loc = finding.source_location
                if loc is not None and loc.file is not None:
                    loc.file = redactor.normalize_path(loc.file, repo=repo)

        # lint_findings defaults to [] in the schema; pass it only when non-empty so a clean scan
        # writes a finding-free artifact via the exclude_unset serializer.
        optional: dict[str, list[LintFinding]] = (
            {"lint_findings": self._findings} if self._findings else {}
        )
        artifact = ClsArtifact(
            schema_version=SCHEMA_VERSION,
            session=self._build_session(),
            **optional,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(to_json(artifact))
        _log.debug(
            "lint_collector.finalize",
            output_path=str(self.output_path),
            findings=len(self._findings),
            redaction_policy=str(self.redaction_policy),
        )
        return self.output_path
