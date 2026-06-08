"""Tests for the ``cl`` console-script dispatcher (``compile_lens.cli``).

Focused on ``cl collect`` (the wired Tool 1 entry): it must drive ``RecompileCollector`` to
write a schema-valid ``.cls.json`` from a real fixture, and reject the modes it can't serve
from a CLI. The other subcommands are still placeholders and are covered minimally.

``main()`` is called in-process (not via subprocess) — it returns an exit code and writes to
the real filesystem, which is all we need to assert; argparse's own exits surface as
``SystemExit``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compile_lens import cli
from compile_lens._schema import from_json

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "recompile"


def test_collect_from_logs_writes_valid_artifact(tmp_path: Path) -> None:
    out = tmp_path / "session.cls.json"
    rc = cli.main(
        ["collect", "--from-logs", str(FIXTURES / "simple_batch_size.log"), "--output", str(out)]
    )

    assert rc == 0
    assert out.exists()
    # Round-trips through the pydantic binding, and the recompile the fixture encodes is present.
    artifact = from_json(out.read_text())
    assert len(artifact.recompilations) == 1


def test_collect_from_tlparse_writes_valid_artifact(tmp_path: Path) -> None:
    out = tmp_path / "session.cls.json"
    rc = cli.main(
        ["collect", "--from-tlparse", str(FIXTURES / "tlparse_output"), "--output", str(out)]
    )

    assert rc == 0
    assert out.exists()
    from_json(out.read_text())  # schema-valid


def test_collect_default_redaction_is_strict(tmp_path: Path) -> None:
    out = tmp_path / "session.cls.json"
    cli.main(
        ["collect", "--from-logs", str(FIXTURES / "simple_batch_size.log"), "--output", str(out)]
    )

    artifact = from_json(out.read_text())
    assert artifact.session.redaction_policy.value == "default-strict"


def test_collect_dynamo_explain_is_programmatic_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["collect", "--from-dynamo-explain", "--output", str(tmp_path / "x.cls.json")])

    assert rc == 2
    assert "programmatic-only" in capsys.readouterr().err


def test_collect_missing_input_returns_io_error(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "collect",
            "--from-logs",
            str(tmp_path / "nope.log"),
            "--output",
            str(tmp_path / "x.cls.json"),
        ]
    )
    assert rc == 3


def test_collect_requires_a_mode(tmp_path: Path) -> None:
    # The mutually-exclusive mode group is required → argparse exits (code 2) before our handler.
    with pytest.raises(SystemExit):
        cli.main(["collect", "--output", str(tmp_path / "x.cls.json")])


def test_unwired_subcommand_reports_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["diff"])
    assert rc == 2
    assert "not implemented yet" in capsys.readouterr().err
