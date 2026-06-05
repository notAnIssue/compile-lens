"""Tests for the Mode A ``TORCH_LOGS=recompiles`` parser (``_logs_parser``).

Drives the parser against the committed fixture corpus (the real artifact stream) plus
small synthetic snippets that stress prefix-format drift and malformed input. The parser
keys off the ``[__recompiles]`` marker and ignores the glog prefix, so the synthetic
"different torch version" snippets vary only the prefix and expect identical extraction.
"""

from __future__ import annotations

from pathlib import Path

from compile_lens._schema import RedactionPolicy, from_json
from compile_lens.collectors._logs_parser import parse_recompiles_log
from compile_lens.collectors.recompile import RecompileCollector

_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "recompile"


def _categories(recs) -> list[str]:
    """The mismatch category word from each recompile's primary guard expression."""
    out = []
    for r in recs:
        expr = (r.failed_guard.expression if r.failed_guard else "") or ""
        out.append(next((w for w in ("size", "dtype", "stride") if w in expr), "other"))
    return out


# ── committed fixtures (the AC1.1 surface) ───────────────────────────────────────────────
def test_parse_simple_batch_size_log() -> None:
    recs = parse_recompiles_log((_FIXTURES / "simple_batch_size.log").read_text())
    assert len(recs) == 1
    (r,) = recs
    assert r.recompilation_id == "0/1"
    assert r.compiled_function_id == "f"
    assert r.trigger_reason == "guard_failure"
    assert r.occurred_at_step == 1
    assert r.failed_guard is not None
    assert "size mismatch at index 0" in r.failed_guard.expression
    assert r.failed_guard.guard_id == "0/0"
    assert r.failed_guard.previous_value == "4"
    assert r.failed_guard.new_value == "8"


def test_parse_mixed_guards() -> None:
    recs = parse_recompiles_log((_FIXTURES / "mixed_guards.log").read_text())
    assert len(recs) == 3
    assert [r.recompilation_id for r in recs] == ["0/1", "0/2", "0/3"]
    # Primary-guard selection (failure against the frame base) yields one distinct
    # category per recompile, matching the mixed_guards oracle.
    assert _categories(recs) == ["size", "dtype", "stride"]


def test_parse_large_storm_all_size() -> None:
    recs = parse_recompiles_log((_FIXTURES / "large_storm.log").read_text())
    assert len(recs) == 49
    assert set(_categories(recs)) == {"size"}
    assert recs[0].recompilation_id == "0/1"
    assert recs[-1].recompilation_id == "0/49"


# ── focused category extraction ──────────────────────────────────────────────────────────
def _one_block(compile_id: str, func: str, guard: str) -> str:
    p = f"V<TS> <PID> torch/_dynamo/guards.py:5188] [{compile_id}] [__recompiles] "
    prior = f"{compile_id.rsplit('/', 1)[0]}/0"
    return (
        f"{p}Recompiling function {func} in /m.py:7\n"
        f"{p}    triggered by the following guard failure(s):\n"
        f"{p}    - {prior}: {guard}\n"
    )


def test_parse_dtype_change() -> None:
    (r,) = parse_recompiles_log(
        _one_block("0/1", "g", "tensor 'x' dtype mismatch. expected Float, actual Double")
    )
    assert r.failed_guard.expression == "tensor 'x' dtype mismatch"
    assert (r.failed_guard.previous_value, r.failed_guard.new_value) == ("Float", "Double")


def test_parse_dynamic_stride() -> None:
    (r,) = parse_recompiles_log(
        _one_block("0/1", "g", "tensor 'x' stride mismatch at index 0. expected 4, actual 1")
    )
    assert r.failed_guard.expression == "tensor 'x' stride mismatch at index 0"
    assert (r.failed_guard.previous_value, r.failed_guard.new_value) == ("4", "1")


def test_guard_without_expected_actual_keeps_full_expression() -> None:
    (r,) = parse_recompiles_log(_one_block("0/1", "g", "L['x'] is not None"))
    assert r.failed_guard.expression == "L['x'] is not None"
    assert r.failed_guard.previous_value is None
    assert r.failed_guard.new_value is None


# ── multi-failure primary selection ──────────────────────────────────────────────────────
def test_multiple_failures_picks_the_base_compile() -> None:
    p = "X <PID> guards.py:1] [0/3] [__recompiles] "
    text = (
        f"{p}Recompiling function f in /m.py:7\n"
        f"{p}    triggered by the following guard failure(s):\n"
        f"{p}    - 0/2: tensor 'x' dtype mismatch. expected Double, actual Float\n"
        f"{p}    - 0/1: tensor 'x' stride mismatch at index 0. expected 4, actual 1\n"
        f"{p}    - 0/0: tensor 'x' stride mismatch at index 0. expected 4, actual 1\n"
    )
    (r,) = parse_recompiles_log(text)
    # The defining failure is against the base 0/0 (stride), not the first-listed 0/2 (dtype).
    assert r.failed_guard.guard_id == "0/0"
    assert "stride mismatch" in r.failed_guard.expression
    assert r.occurred_at_step == 3


# ── prefix-format robustness (cross torch version) ───────────────────────────────────────
def test_parse_pytorch_25_format() -> None:
    # A distributed-rank prefix shape; body identical.
    line_prefix = (
        "[rank0]:V0101 12:00:00.000000 100 torch/_dynamo/guards.py:1234] [0/1] [__recompiles] "
    )
    text = (
        f"{line_prefix}Recompiling function g in /a/b.py:10\n"
        f"{line_prefix}    - 0/0: tensor 'y' size mismatch at index 1. expected 16, actual 32\n"
    )
    (r,) = parse_recompiles_log(text)
    assert r.recompilation_id == "0/1"
    assert r.compiled_function_id == "g"
    assert (r.failed_guard.previous_value, r.failed_guard.new_value) == ("16", "32")


def test_parse_pytorch_nightly_format() -> None:
    # A differently-shaped timestamp / source ref; body identical.
    line_prefix = "V1231 23:59:59.999999 9 torch/_dynamo/guards.py:9999] [0/2] [__recompiles] "
    text = (
        f"{line_prefix}Recompiling function h in /x/y.py:5\n"
        f"{line_prefix}    - 0/0: tensor 'z' dtype mismatch. expected BFloat16, actual Float\n"
    )
    (r,) = parse_recompiles_log(text)
    assert r.recompilation_id == "0/2"
    assert r.compiled_function_id == "h"
    assert r.failed_guard.new_value == "Float"


# ── robustness / edge cases ──────────────────────────────────────────────────────────────
def test_parse_malformed_line_skips_gracefully() -> None:
    p = "V<TS> <PID> guards.py:1] [0/1] [__recompiles] "
    text = (
        f"{p}Recompiling function f in /m.py:7\n"
        f"{p}this is a malformed recompiles line that matches no pattern\n"
        f"{p}    - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8\n"
    )
    recs = parse_recompiles_log(text)  # must not raise
    assert len(recs) == 1
    assert recs[0].failed_guard.expression == "tensor 'x' size mismatch at index 0"


def test_parse_empty_text_returns_empty() -> None:
    assert parse_recompiles_log("") == []


def test_parse_non_artifact_lines_ignored() -> None:
    text = "some unrelated stdout\nWARNING: numpy failed to import\n[INFO] torch: hello\n"
    assert parse_recompiles_log(text) == []


# ── collector wiring (Mode A end-to-end into a .cls.json) ─────────────────────────────────
def test_from_logs_ingests_into_valid_cls_json(tmp_path: Path) -> None:
    out = tmp_path / "session.cls.json"
    c = RecompileCollector(
        out,
        session_id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-01-01T00:00:00Z",
        torch_version="2.6.0",
        redaction_policy=RedactionPolicy.DEFAULT_STRICT,
    )
    c.from_logs(_FIXTURES / "mixed_guards.log")
    artifact = from_json(c.finalize().read_text())
    assert len(artifact.recompilations) == 3
    assert [r.recompilation_id for r in artifact.recompilations] == ["0/1", "0/2", "0/3"]
