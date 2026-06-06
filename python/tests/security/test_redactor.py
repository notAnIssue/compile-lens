"""Tests for collect-time redaction (``compile_lens.security.redactor``) + its wiring into
``RecompileCollector`` (discipline D11). Unit tests pin each primitive; the collector tests
pin that a strict-policy artifact never holds a raw secret and refuses a credential path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compile_lens._schema import CompiledGraph, RedactionPolicy, from_json
from compile_lens.collectors.recompile import RecompileCollector
from compile_lens.security.redactor import (
    SensitivePathError,
    filter_env_vars,
    hash_host,
    normalize_path,
    scrub_command,
)


# ── path normalization (§2.1) ────────────────────────────────────────────────────────────
def test_path_normalization_home() -> None:
    assert normalize_path("/home/jdoe/work/myrepo/model.py", repo="myrepo") == "{repo}/model.py"
    # macOS variant + no intermediate dir
    assert normalize_path("/Users/jdoe/myrepo/m.py", repo="myrepo") == "{repo}/m.py"


def test_path_normalization_conda_env() -> None:
    p = "/opt/conda/envs/ml/lib/python3.11/site-packages/torch/_inductor/foo.py"
    assert normalize_path(p) == "{torch_install}/_inductor/foo.py"


def test_ssh_path_refused() -> None:
    with pytest.raises(SensitivePathError) as exc:
        normalize_path("/home/jdoe/.ssh/id_rsa")
    assert exc.value.code == "CLS-E0008"
    assert exc.value.path == "/home/jdoe/.ssh/id_rsa"


def test_aws_and_gnupg_paths_refused() -> None:
    for p in ("/home/u/.aws/credentials", "/home/u/.gnupg/secring.gpg"):
        with pytest.raises(SensitivePathError):
            normalize_path(p)


# ── argv token scrub (§2.2) ──────────────────────────────────────────────────────────────
def test_argv_hf_token_scrubbed() -> None:
    out = scrub_command("python train.py --hf-token=hf_abcdefghij0123456789xyz")
    assert "--hf-token=<scrubbed>" in out
    assert "hf_abcdefghij" not in out


def test_argv_openai_key_scrubbed() -> None:
    # both the --api-key flag form and a bare sk- token
    sk = "sk-abc0123456789def0123456789"
    assert scrub_command(f"run --api-key={sk}") == "run --api-key=<scrubbed>"
    assert sk not in scrub_command(f"x {sk}")


def test_argv_bearer_and_aws_scrubbed() -> None:
    assert "Bearer <scrubbed>" in scrub_command("curl -H Authorization: Bearer ab.cd-ef_123")
    assert "AKIA<scrubbed>" in scrub_command("aws --key AKIAIOSFODNN7EXAMPLE")


# ── FQDN hash (§2.5) ─────────────────────────────────────────────────────────────────────
def test_fqdn_hashed() -> None:
    h = hash_host("gpu-node-42.cluster.example.com", salt="fixed-salt")
    assert h.startswith("dh-")
    assert len(h) == len("dh-") + 8
    assert "example.com" not in h
    # stable for same (fqdn, salt); different fqdn -> different hash
    assert h == hash_host("gpu-node-42.cluster.example.com", salt="fixed-salt")
    assert h != hash_host("other-node.example.com", salt="fixed-salt")


# ── env var filtering (§2.6) ─────────────────────────────────────────────────────────────
def test_env_var_whitelist() -> None:
    out = filter_env_vars(
        {"TORCH_LOGS": "+recompiles", "TORCHINDUCTOR_CACHE_DIR": "/x", "PATH": "/usr/bin"}
    )
    assert out == {"TORCH_LOGS": "+recompiles", "TORCHINDUCTOR_CACHE_DIR": "/x"}


def test_env_var_denylist_blocks_keys() -> None:
    out = filter_env_vars({"HF_TOKEN": "secret", "TORCH_API_KEY": "secret", "NCCL_DEBUG": "INFO"})
    # HF_TOKEN (deny-exact) + TORCH_API_KEY (_KEY suffix beats the TORCH_ whitelist) dropped
    assert out == {"NCCL_DEBUG": "INFO"}


# ── collector wiring (D11 at collect time) ───────────────────────────────────────────────
def _make(tmp_path: Path, **kw: object) -> RecompileCollector:
    base = {
        "session_id": "00000000-0000-4000-8000-000000000000",
        "timestamp": "2026-01-01T00:00:00Z",
        "torch_version": "2.6.0",
    }
    return RecompileCollector(tmp_path / "s.cls.json", **{**base, **kw})  # type: ignore[arg-type]


def test_command_scrubbed_in_strict_artifact(tmp_path: Path) -> None:
    c = _make(tmp_path, command="python train.py --hf-token=hf_abcdefghij0123456789xyz")
    text = c.finalize().read_text()
    assert "hf_abcdefghij" not in text
    assert from_json(text).session.command == "python train.py --hf-token=<scrubbed>"


def test_internal_policy_keeps_raw_command(tmp_path: Path) -> None:
    c = _make(
        tmp_path,
        redaction_policy=RedactionPolicy.INTERNAL,
        command="python train.py --hf-token=hf_abcdefghij0123456789xyz",
    )
    # internal is a trusted-team policy: the collector does not auto-scrub.
    command = from_json(c.finalize().read_text()).session.command
    assert command is not None and command.endswith("hf_abcdefghij0123456789xyz")


def test_host_hashed_in_strict_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLS_INSTALL_ID", "deterministic-test-salt")
    c = _make(tmp_path, host="gpu-node-42.cluster.example.com")
    host = from_json(c.finalize().read_text()).session.host
    assert host is not None and host.startswith("dh-")
    assert "example.com" not in host


def test_sensitive_compiled_graph_path_refused(tmp_path: Path) -> None:
    c = _make(tmp_path)
    c.add_records(
        compiled_graphs=[CompiledGraph(graph_id="g1", fx_graph_path="/home/u/.ssh/id_rsa")]
    )
    with pytest.raises(SensitivePathError):
        c.finalize()
