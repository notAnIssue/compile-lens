"""Read the correctness database (TOML) on the Python side.

The Rust ``cls-correctness-db`` crate is the canonical loader — it gives the analyzer each pattern's
evidence and severity. This is the **Python half** of the same file (ADR-036): the Layer-1 front-end
reads each operator-family pattern's ``[detector]`` rule to configure the scanner, and the fixture
test reads each pattern's fixtures. The two languages meet on the one TOML file (ADR-006, no FFI);
Python reads only the detection-side fields it needs.

``tomllib`` is standard library on the project's baseline (Python >= 3.11).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detector:
    """An operator-family detection rule: the watched operator and its watched parameters."""

    operator: str
    params: tuple[str, ...]


@dataclass(frozen=True)
class DbPattern:
    """The detection-side view of one database pattern (name, fixtures, optional detector).

    Evidence fields (issue, workaround, …) are deliberately not read here — they belong to the Rust
    analyzer's view of the same file."""

    name: str
    fixture_positive: str
    fixture_negative: str
    detector: Detector | None


def load_patterns(db_path: Path | str) -> list[DbPattern]:
    """Parse the correctness database TOML into the detection-side patterns, in file order."""
    data = tomllib.loads(Path(db_path).read_text())
    patterns: list[DbPattern] = []
    for entry in data.get("patterns", []):
        raw = entry.get("detector")
        detector = Detector(raw["operator"], tuple(raw["params"])) if raw is not None else None
        fixtures = entry["fixtures"]
        patterns.append(
            DbPattern(
                name=entry["name"],
                fixture_positive=fixtures["positive"],
                fixture_negative=fixtures["negative"],
                detector=detector,
            )
        )
    return patterns


def operator_rules(patterns: list[DbPattern]) -> dict[str, dict[str, str]]:
    """Build the scanner's ``operator_rules`` — ``operator -> {param: pattern_name}`` (ADR-036).

    Each operator-family pattern contributes its name as the category to emit when any of its
    watched parameters is seen on its operator. Structural patterns (no detector) are skipped — the
    scanner detects those without a rule.
    """
    rules: dict[str, dict[str, str]] = {}
    for pattern in patterns:
        if pattern.detector is None:
            continue
        params = rules.setdefault(pattern.detector.operator, {})
        for param in pattern.detector.params:
            params[param] = pattern.name
    return rules


def load_operator_rules(db_path: Path | str) -> dict[str, dict[str, str]]:
    """Convenience: parse the database and build the scanner's ``operator_rules`` in one call."""
    return operator_rules(load_patterns(db_path))
