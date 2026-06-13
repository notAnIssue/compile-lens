"""Tests for compile_lens.correctness_db — the Python-side reader of the correctness database."""

from __future__ import annotations

from pathlib import Path

from compile_lens.correctness_db import load_operator_rules, load_patterns

# One structural pattern (no detector) and one operator-family pattern (with a detector).
_DB = """
[[patterns]]
name = "struct_one"
minimal_repro = "..."
pytorch_issue = "https://github.com/pytorch/pytorch/issues/1"
workaround = "..."
[patterns.fixtures]
positive = "POS1"
negative = "NEG1"

[[patterns]]
name = "op_one"
minimal_repro = "..."
pytorch_issue = "https://github.com/pytorch/pytorch/issues/2"
workaround = "..."
[patterns.detector]
operator = "diag_embed"
params = ["dim1", "dim2"]
[patterns.fixtures]
positive = "POS2"
negative = "NEG2"
"""


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "db.toml"
    path.write_text(_DB)
    return path


def test_load_patterns_reads_name_fixtures_and_detector(tmp_path: Path) -> None:
    struct, op = load_patterns(_db(tmp_path))
    assert struct.name == "struct_one"
    assert struct.detector is None  # structural: no detection rule
    assert (struct.fixture_positive, struct.fixture_negative) == ("POS1", "NEG1")
    assert op.detector is not None
    assert op.detector.operator == "diag_embed"
    assert op.detector.params == ("dim1", "dim2")


def test_operator_rules_maps_each_watched_param_to_the_pattern_name(tmp_path: Path) -> None:
    # The structural pattern contributes nothing; the operator pattern maps each of its watched
    # params to its own name (the category the scanner will emit) — ADR-036.
    rules = load_operator_rules(_db(tmp_path))
    assert rules == {"diag_embed": {"dim1": "op_one", "dim2": "op_one"}}


def test_structural_only_database_yields_no_operator_rules(tmp_path: Path) -> None:
    path = tmp_path / "s.toml"
    path.write_text(
        '[[patterns]]\nname = "s"\nminimal_repro = "x"\n'
        'pytorch_issue = "u"\nworkaround = "w"\n'
        '[patterns.fixtures]\npositive = "p"\nnegative = "n"\n'
    )
    assert load_operator_rules(path) == {}
