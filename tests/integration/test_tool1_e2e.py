"""End-to-end integration tests for Tool 1 (recompile aggregator).

Drives the full pipeline the way the toolkit actually runs it, across the language boundary:

    fixture log / tlparse dir  --(Python collector)-->  .cls.json
    .cls.json                  --(Rust `cl recompile-summary`)-->  findings JSON

then asserts the rendered findings against the committed per-fixture oracle
(`<fixture>.expected.json`). The collector runs as a Python API call (Mode A/B parse text,
no torch needed); the analyzer runs as the built `cl` binary via subprocess — the same
file + subprocess control-flow real use takes (the boundary is a JSON file, not FFI).

Scope note: the oracles' source-attribution fields and *live* cross-PyTorch-version capture
(2.5 / 2.6 / nightly) are not exercised here — attribution isn't surfaced in Tool 1's findings
yet, and the fixtures are version-agnostic captured text. The live multi-version matrix is the
Phase-1 DoD CI concern, not fakeable from static fixtures.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from compile_lens.collectors.recompile import RecompileCollector

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "recompile"

# Deterministic session metadata so the collected artifact is byte-stable (mirrors the
# collector unit tests + the determinism discipline the fixture corpus relies on).
_SESSION_KW = {
    "session_id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.6.0",
}

# Mode-A fixtures, each with a committed semantic oracle (`<name>.expected.json`).
MODE_A_CASES = ["simple_batch_size", "mixed_guards", "nonshape_guards", "large_storm"]


@pytest.fixture(scope="session", autouse=True)
def _build_cl() -> None:
    """Build the `cl` binary once so per-test subprocess calls are fast."""
    subprocess.run(
        ["cargo", "build", "--quiet", "--package", "cls-cli", "--bin", "cl"],
        cwd=REPO_ROOT,
        check=True,
    )


def _recompile_summary_json(session: Path) -> dict:
    """Run `cl recompile-summary <session> --format json` and parse stdout."""
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--package",
            "cls-cli",
            "--bin",
            "cl",
            "--",
            "recompile-summary",
            str(session),
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


@pytest.mark.parametrize("case", MODE_A_CASES)
def test_mode_a_pipeline_matches_oracle(case: str, tmp_path: Path) -> None:
    out = tmp_path / f"{case}.cls.json"
    collector = RecompileCollector(out, **_SESSION_KW)  # type: ignore[arg-type]
    collector.from_logs(FIXTURES / f"{case}.log")
    collector.finalize()

    artifact = json.loads(out.read_text())
    findings = _recompile_summary_json(out)
    oracle = json.loads((FIXTURES / f"{case}.expected.json").read_text())

    # 1. Recompile count: collect + analyze agree with the oracle.
    assert findings["total_recompilations"] == oracle["expected_recompile_count"], (
        f"{case}: recompile count mismatch"
    )

    categories = {c["category"] for c in findings["guard_categories"]}
    suggestion_text = " ".join(s["text"] for s in findings["top_suggestions"])
    # Raw guard text comes from the *collected* artifact (the analyzer surfaces a canonical
    # template + axis, not the raw expression), so guard_text_contains is checked there.
    guard_exprs = " ".join(
        (r.get("failed_guard") or {}).get("expression", "") or ""
        for r in artifact.get("recompilations", [])
    )

    # 2. Each expected cluster: category surfaced, raw guard captured, suggestion advises right.
    for expected in oracle["expected_clusters"]:
        assert expected["category"] in categories, (
            f"{case}: expected a {expected['category']} cluster, got {sorted(categories)}"
        )
        for needle in expected.get("guard_text_contains", []):
            assert needle in guard_exprs, f"{case}: collected guard text missing {needle!r}"
        for keyword in expected.get("expected_suggestion_keywords", []):
            assert keyword in suggestion_text, f"{case}: no suggestion mentioned {keyword!r}"


def test_mode_b_tlparse_pipeline_runs(tmp_path: Path) -> None:
    """Mode B (tlparse dir) -> analyze -> render completes and yields valid findings.

    No oracle: the tlparse fixture's recompiles carry no `failed_guard`, so this asserts the
    cross-format pipeline is wired end-to-end (not specific clusters).
    """
    out = tmp_path / "tlparse.cls.json"
    collector = RecompileCollector(out, **_SESSION_KW)  # type: ignore[arg-type]
    collector.from_tlparse(FIXTURES / "tlparse_output")
    collector.finalize()

    findings = _recompile_summary_json(out)
    assert isinstance(findings["total_recompilations"], int)
    assert findings["total_recompilations"] >= 0
    assert isinstance(findings["guard_categories"], list)
