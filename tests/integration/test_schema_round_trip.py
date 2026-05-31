"""Cross-language schema round-trip tests.

Proves the Rust (serde, `cls-schema`) and Python (pydantic, `compile_lens._schema`)
bindings agree on schema v0.5.0: an artifact survives Python -> Rust -> Python (and the
reverse) with its fields intact, and each side serializes deterministically.

The Rust side is exercised through the `cls-schema` `roundtrip` example via subprocess —
the same file + subprocess control-flow the real toolkit uses (design.md ADR-006: data
crosses the boundary only as a JSON file, no FFI).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from compile_lens._schema import from_json, to_json

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "schema" / "examples"
CASES = ["minimal", "full"]


def _rust_roundtrip(src: Path, dst: Path) -> None:
    """Read `src`, deserialize+reserialize through the Rust harness, write `dst`."""
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--package",
            "cls-schema",
            "--example",
            "roundtrip",
            "--",
            str(src),
            str(dst),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


@pytest.fixture(scope="session", autouse=True)
def _build_harness() -> None:
    """Build the Rust harness once up front so the per-test subprocess calls are fast."""
    subprocess.run(
        ["cargo", "build", "--quiet", "--package", "cls-schema", "--example", "roundtrip"],
        cwd=REPO_ROOT,
        check=True,
    )


@pytest.mark.parametrize("case", CASES)
def test_python_to_rust_to_python_fields_equal(case: str, tmp_path: Path) -> None:
    """Python writes -> Rust reads+rewrites -> Python reads back == original (semantic)."""
    original = from_json((EXAMPLES / f"{case}.cls.json").read_text())
    py_out = tmp_path / "py.json"
    py_out.write_text(to_json(original))
    rust_out = tmp_path / "rust.json"
    _rust_roundtrip(py_out, rust_out)
    assert from_json(rust_out.read_text()) == original


@pytest.mark.parametrize("case", CASES)
def test_rust_reads_example_python_agrees(case: str, tmp_path: Path) -> None:
    """Python reading Rust's output of the raw example == Python reading the example."""
    src = EXAMPLES / f"{case}.cls.json"
    rust_out = tmp_path / "rust.json"
    _rust_roundtrip(src, rust_out)
    assert from_json(rust_out.read_text()) == from_json(src.read_text())


@pytest.mark.parametrize("case", CASES)
def test_rust_to_python_to_rust_byte_identical(case: str, tmp_path: Path) -> None:
    """Rust -> Python -> Rust: the Rust-emitted form must not drift through Python."""
    src = EXAMPLES / f"{case}.cls.json"
    r1 = tmp_path / "r1.json"
    _rust_roundtrip(src, r1)
    p = tmp_path / "p.json"
    p.write_text(to_json(from_json(r1.read_text())))
    r2 = tmp_path / "r2.json"
    _rust_roundtrip(p, r2)
    assert r1.read_bytes() == r2.read_bytes()


@pytest.mark.parametrize("case", CASES)
def test_rust_determinism_byte_identical(case: str, tmp_path: Path) -> None:
    """Same input through the Rust harness twice -> byte-identical output (D10)."""
    src = EXAMPLES / f"{case}.cls.json"
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _rust_roundtrip(src, a)
    _rust_roundtrip(src, b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("case", CASES)
def test_python_determinism_byte_identical(case: str) -> None:
    """Python serialization is deterministic (D10)."""
    model = from_json((EXAMPLES / f"{case}.cls.json").read_text())
    assert to_json(model) == to_json(model)


def test_unknown_field_survives_cross_language(tmp_path: Path) -> None:
    """ADR-027 / DR4 parity: an unknown key at a growth point survives Python -> Rust -> Python."""
    data = json.loads((EXAMPLES / "full.cls.json").read_text())
    data["future_top_level"] = {"x": 1}
    data["session"]["gpu_arch"] = "hopper"
    py_out = tmp_path / "py.json"
    py_out.write_text(json.dumps(data))
    rust_out = tmp_path / "rust.json"
    _rust_roundtrip(py_out, rust_out)

    # pydantic captures unknowns in `model_extra` (the Rust side names the field `extra`);
    # same wire behavior (re-emitted at the same level), different access API.
    back = from_json(rust_out.read_text())
    assert (back.model_extra or {}).get("future_top_level") == {"x": 1}
    assert (back.session.model_extra or {}).get("gpu_arch") == "hopper"
