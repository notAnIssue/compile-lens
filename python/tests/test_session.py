"""Tests for ``cl.session()`` — the P7 source-of-truth hero entry (ADR-026).

The recompile capture is exercised two ways: by injecting a synthetic ``[__recompiles]`` stream on
the stderr fd (no torch needed — proves the fd-tee + parser wiring), and end-to-end against a real
``torch.compile`` that recompiles (proves the whole path).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compile_lens import Session, session

# A minimal synthetic TORCH_LOGS=recompiles block (the format parse_recompiles_log consumes).
_FAKE_RECOMPILE = (
    b"V0616 12:00:00.0 1 torch/_dynamo/guards.py:1] [0/1] [__recompiles] "
    b"Recompiling function f in model.py:42\n"
    b"V0616 12:00:00.0 1 torch/_dynamo/guards.py:1] [0/1] [__recompiles]     "
    b"triggered by the following guard failure(s):\n"
    b"V0616 12:00:00.0 1 torch/_dynamo/guards.py:1] [0/1] [__recompiles]     "
    b"- 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8\n"
)


def test_session_is_a_context_manager(tmp_path: Path) -> None:
    with session(output_dir=tmp_path) as s:
        assert isinstance(s, Session)
    assert s.artifact_path is not None and s.artifact_path.exists()


def test_default_probes_are_recompile() -> None:
    # ADR-026's frozen default also names `iterations`; until that probe is built the working
    # default is just `recompile`, so the six-line example runs.
    assert session().probes == frozenset({"recompile"})


def test_unknown_probe_is_value_error() -> None:
    with pytest.raises(ValueError, match="unknown probe"):
        session(probes={"bogus"})


def test_unbuilt_probe_is_not_implemented() -> None:
    # `iterations` is a recognized ADR-026 name but unbuilt -> NotImplementedError, distinct from
    # the ValueError an unknown name raises.
    with pytest.raises(NotImplementedError, match="iterations"):
        session(probes={"recompile", "iterations"})


def test_session_writes_valid_cls_json_with_captured_recompile(tmp_path: Path) -> None:
    with session(output_dir=tmp_path) as s:
        os.write(2, _FAKE_RECOMPILE)  # straight to the stderr fd the session tees
    data = json.loads(s.artifact_path.read_text())  # type: ignore[union-attr]
    assert data["schema_version"] == "0.5.0"
    assert len(data.get("recompilations", [])) == 1
    rec = data["recompilations"][0]
    assert rec["compiled_function_id"] == "f"
    assert rec["recompilation_id"] == "0/1"


def test_empty_session_still_writes_a_valid_artifact(tmp_path: Path) -> None:
    with session(output_dir=tmp_path):
        pass
    # No recompiles captured -> a schema-valid artifact with no recompilations, not a crash.
    data = json.loads((tmp_path / "session.cls.json").read_text())
    assert data["schema_version"] == "0.5.0"
    assert data.get("recompilations", []) == []


def test_redaction_applied_under_default_strict(tmp_path: Path) -> None:
    with session(output_dir=tmp_path) as s:
        pass
    data = json.loads(s.artifact_path.read_text())  # type: ignore[union-attr]
    assert data["session"]["redaction_policy"] == "default-strict"
    # Strict policy replaces the host FQDN with a stable dh-<hash> identifier (D11).
    assert data["session"]["host"].startswith("dh-")


def test_report_is_not_built_yet(tmp_path: Path) -> None:
    with session(output_dir=tmp_path) as s:
        pass
    with pytest.raises(NotImplementedError, match="cl session report"):
        s.report()


def test_session_captures_a_real_torch_recompile(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    torch = pytest.importorskip("torch")
    # The session captures the stderr fd; pytest normally redirects it and replaces sys.stderr, so
    # torch's log would bypass the fd. `capfd.disabled()` restores the real fds for this block (the
    # condition a real user's script is always in). Reset torch's recompiles log first so its
    # handler re-binds to the restored stderr rather than a prior test's capture stream.
    with capfd.disabled():
        torch._logging.set_logs(recompiles=False)
        with session(output_dir=tmp_path) as s:

            @torch.compile
            def f(x):  # type: ignore[no-untyped-def]
                return x.sum() + 1

            for n in (4, 8, 16):  # a changing shape forces a recompile (static by default)
                f(torch.randn(n))

    data = json.loads(s.artifact_path.read_text())  # type: ignore[union-attr]
    assert len(data.get("recompilations", [])) >= 1
