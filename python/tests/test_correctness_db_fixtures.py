"""Behavioral validation of every production correctness-database pattern's fixtures (ADR-036).

For each pattern in ``data/correctness_db.toml``: scanning its **positive** fixture must surface its
own category, and scanning its **negative** fixture must not. Detection lives on the Python scanner
side, so this is the behavioral half of the database gate — the Rust ``pattern_fixtures`` test
covers structural integrity (four items, real issue URL, unique names, well-formed detectors).

The scanner is configured from the database's own detector rules, so this also exercises the full
DB-driven granular path: a pattern's positive fixture is detected only if its detector rule (or the
built-in structural logic) actually fires and emits the pattern's name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compile_lens.collectors.lint import LintPatternScanner
from compile_lens.correctness_db import DbPattern, load_operator_rules, load_patterns

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "correctness_db.toml"

_PATTERNS = load_patterns(DB_PATH)
_RULES = load_operator_rules(DB_PATH)


def _categories(source: str) -> set[str]:
    return {hit.pattern_name for hit in LintPatternScanner(_RULES).scan(source)}


def test_database_is_non_empty() -> None:
    assert _PATTERNS, "no patterns loaded from the production database"


@pytest.mark.parametrize("pattern", _PATTERNS, ids=lambda p: p.name)
def test_positive_fixture_is_detected(pattern: DbPattern) -> None:
    assert pattern.name in _categories(pattern.fixture_positive)


@pytest.mark.parametrize("pattern", _PATTERNS, ids=lambda p: p.name)
def test_negative_fixture_is_not_detected(pattern: DbPattern) -> None:
    assert pattern.name not in _categories(pattern.fixture_negative)
