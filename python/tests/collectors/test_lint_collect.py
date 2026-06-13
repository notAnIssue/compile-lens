"""Tests for compile_lens.collectors.lint_collect.

The collector drives the Layer-1 AST scanner over source files and writes the candidate findings
into a schema-valid ``.cls.json``. No torch is needed — the scanner is pure ``ast`` — and session
metadata is pinned for deterministic output. The Rust ``cl compile-lint`` is what assigns the real
severity; here every candidate is a pre-analysis placeholder.
"""

from __future__ import annotations

from pathlib import Path

from compile_lens._schema import from_json
from compile_lens.collectors.lint_collect import LintCollector

# `expand` returns an alias; the indexed write through it is the in_place_op_on_alias anti-pattern.
ALIAS_SRC = "y = x.expand(2, 3)\ny[0] = 1\n"
CLEAN_SRC = "y = x.clone()\ny[0] = 1\n"  # write through a clone is safe -> no finding


def _collector(out: Path) -> LintCollector:
    return LintCollector(
        out,
        session_id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-06-13T00:00:00Z",
        torch_version="2.6.0",
    )


def test_in_place_alias_becomes_a_candidate_finding(tmp_path: Path) -> None:
    src = tmp_path / "model.py"
    src.write_text(ALIAS_SRC)
    out = tmp_path / "out.cls.json"

    collector = _collector(out)
    added = collector.scan_path(src)
    collector.finalize()

    assert added == 1
    artifact = from_json(out.read_text())
    assert len(artifact.lint_findings) == 1
    finding = artifact.lint_findings[0]
    assert finding.pattern_category == "in_place_op_on_alias"
    assert finding.finding_id == "lf_001"
    # Layer-1 only: a pre-analysis placeholder severity and an AST-match confidence.
    assert finding.severity == "candidate"
    assert finding.confidence == "ast_match"
    assert finding.source_location is not None
    assert finding.source_location.line_start == 2


def test_finding_ids_are_positional(tmp_path: Path) -> None:
    # Two independent aliases, each written through -> two findings, numbered in scan order.
    src = tmp_path / "model.py"
    src.write_text("a = x.expand(2, 3)\na[0] = 1\nb = x.view(6)\nb[0] = 2\n")
    out = tmp_path / "out.cls.json"

    collector = _collector(out)
    collector.scan_path(src)
    collector.finalize()

    ids = [f.finding_id for f in from_json(out.read_text()).lint_findings]
    assert ids == ["lf_001", "lf_002"]


def test_scans_a_directory_recursively(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text(ALIAS_SRC)
    (pkg / "sub" / "b.py").write_text(ALIAS_SRC)
    out = tmp_path / "out.cls.json"

    collector = _collector(out)
    added = collector.scan_path(pkg)
    collector.finalize()

    assert added == 2
    files = {f.source_location.file for f in from_json(out.read_text()).lint_findings}
    assert files == {str(pkg / "a.py"), str(pkg / "sub" / "b.py")}


def test_clean_source_writes_a_finding_free_artifact(tmp_path: Path) -> None:
    src = tmp_path / "model.py"
    src.write_text(CLEAN_SRC)
    out = tmp_path / "out.cls.json"

    collector = _collector(out)
    added = collector.scan_path(src)
    collector.finalize()

    assert added == 0
    # A clean scan still writes a schema-valid artifact; it just has no findings.
    assert from_json(out.read_text()).lint_findings == []


def test_strict_keeps_the_relative_source_path(tmp_path: Path, monkeypatch) -> None:
    # A project-relative path is the useful part of a finding (where the bug is); strict redaction
    # only collapses machine-absolute prefixes, so a relative path survives unchanged.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "model.py").write_text(ALIAS_SRC)
    out = tmp_path / "out.cls.json"

    collector = _collector(out)
    collector.scan_path("pkg/model.py")
    collector.finalize()

    artifact = from_json(out.read_text())
    assert artifact.session.redaction_policy.value == "default-strict"
    assert artifact.lint_findings[0].source_location.file == "pkg/model.py"


# An in-place-on-alias write, silenced by an inline suppression comment.
SUPPRESSED_SRC = "y = x.expand(2, 3)\ny[0] = 1  # compile-lint: ignore[in_place_op_on_alias]\n"


def test_suppression_comment_is_honored_through_the_collector(tmp_path: Path) -> None:
    # The scanner applies suppression inside scan(), so a suppressed finding must never reach the
    # written artifact — suppression works end-to-end through the collector, not just in unit tests.
    src = tmp_path / "model.py"
    src.write_text(SUPPRESSED_SRC)
    out = tmp_path / "out.cls.json"

    collector = _collector(out)
    added = collector.scan_path(src)
    collector.finalize()

    assert added == 0
    assert from_json(out.read_text()).lint_findings == []
