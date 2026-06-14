"""Test-first spec for the Tool 5 ``KernelMetadataCollector``.

These tests ARE the contract. The implementation that satisfies them lives in
``compile_lens/collectors/kernel_metadata.py``. What this PR pins:

  * collector construction + a fail-closed redaction default (D11);
  * ``add_kernel`` assembling one ``kernels[]`` record, splitting responsibilities:
    the **compiled-kernel object** (a duck-typed Triton ``CompiledKernel``) supplies the
    compile-product facts (``num_regs`` / ``n_spills`` / ``num_warps`` / ``num_stages`` /
    shared memory), while the **caller** supplies the launch + semantic facts that are not in
    a kernel's static metadata (``grid`` / ``block`` / ``flops`` / ``bytes_*``);
  * ``kernel_source_excerpt`` omitted by default, and refused under a strict policy even when
    the caller opts in (fail-closed);
  * path normalization at ``finalize`` under a strict policy (D11);
  * ``finalize()`` writing a schema-valid ``.cls.json``.

Explicitly **out of scope** here (later sections):

  * Measured timings (``measurements``) — that is the proton adapter (a later change).
  * The roofline model itself (``roofline_predictions[]``) — Rust, already landed.

No GPU/Triton is required: the compiled kernel is duck-typed, so a recorded/fake object with
the same attributes drives the extraction (mirrors how ``test_artifact`` duck-types FX nodes).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from compile_lens._schema import RedactionPolicy, from_json
from compile_lens.collectors.kernel_metadata import KernelMetadataCollector

# repo root = parents[3] of python/tests/collectors/test_kernel_metadata.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = json.loads((_REPO_ROOT / "schema" / "v0.5.0.json").read_text())

# Deterministic session metadata so finalize() output is byte-stable across machines.
_SESSION_KW = {
    "session_id": "00000000-0000-4000-8000-000000000000",
    "timestamp": "2026-01-01T00:00:00Z",
    "torch_version": "2.6.0",
}


def _make(tmp_path: Path, **overrides: object) -> KernelMetadataCollector:
    out = tmp_path / "session.cls.json"
    return KernelMetadataCollector(out, **{**_SESSION_KW, **overrides})  # type: ignore[arg-type]


def _fake_compiled_kernel(**overrides: object) -> SimpleNamespace:
    """A duck-typed stand-in for a Triton ``CompiledKernel``.

    Triton exposes ``n_regs`` / ``n_spills`` as attributes and ``num_warps`` / ``num_stages`` /
    ``shared`` on a ``.metadata`` object; we mirror that shape so the extractor can be exercised
    without a GPU. ``overrides`` patch the top-level attributes for the degradation tests.
    """
    meta = SimpleNamespace(num_warps=4, num_stages=3, shared=8192)
    base = {"n_regs": 40, "n_spills": 0, "metadata": meta, "asm": {"ptx": ".entry k() { ret; }"}}
    base.update(overrides)
    return SimpleNamespace(**base)


# ── construction ────────────────────────────────────────────────────────────────────────
def test_collector_init(tmp_path: Path) -> None:
    c = _make(tmp_path)
    # Fail-closed default: absent an explicit choice, the strictest policy applies (D11).
    assert c.redaction_policy is RedactionPolicy.DEFAULT_STRICT
    assert c.output_path == tmp_path / "session.cls.json"


def test_unknown_redaction_policy_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make(tmp_path, redaction_policy="totally-made-up")


# ── extraction from a (duck-typed) Triton kernel ──────────────────────────────────────────
def test_extracts_compile_product_facts_from_kernel(tmp_path: Path) -> None:
    c = _make(tmp_path)
    c.add_kernel("k0", "triton_poi_fused_add_0", compiled_kernel=_fake_compiled_kernel())
    k = c.kernels[0]

    # num_regs / n_spills are features (the register-pressure proxy Layer 2 reads).
    assert k.features is not None
    assert k.features.num_regs == 40
    assert k.features.n_spills == 0
    # num_warps / num_stages / shared come off the kernel's metadata into launch_config.
    assert k.launch_config is not None
    assert k.launch_config.num_warps == 4
    assert k.launch_config.num_stages == 3
    assert k.launch_config.shared_mem_bytes == 8192


def test_caller_supplies_launch_and_semantic_facts(tmp_path: Path) -> None:
    # grid/block (launch) and flops/bytes (kernel semantics) are NOT in a kernel's static
    # metadata, so the caller provides them; they must land in the record.
    c = _make(tmp_path)
    c.add_kernel(
        "k0",
        "triton_red_fused_sum_1",
        compiled_kernel=_fake_compiled_kernel(),
        grid=[128, 1, 1],
        block=[256, 1, 1],
        flops=1.2e10,
        bytes_loaded=4.0e7,
        bytes_stored=2.0e7,
    )
    k = c.kernels[0]
    assert k.launch_config is not None
    assert k.launch_config.grid == [128, 1, 1]
    assert k.launch_config.block == [256, 1, 1]
    assert k.features is not None
    assert k.features.flops == pytest.approx(1.2e10)
    assert k.features.bytes_loaded == pytest.approx(4.0e7)
    assert k.features.bytes_stored == pytest.approx(2.0e7)


def test_all_features_present_when_fully_specified(tmp_path: Path) -> None:
    # AC: every feature (incl. num_regs / n_spills) is captured.
    c = _make(tmp_path)
    c.add_kernel(
        "k0",
        "triton_poi_fused_mul_2",
        compiled_kernel=_fake_compiled_kernel(n_regs=128, n_spills=12),
        block=[128, 1, 1],
        flops=5.0e9,
        bytes_loaded=1.0e7,
        bytes_stored=1.0e7,
    )
    f = c.kernels[0].features
    assert f is not None
    assert all(
        v is not None
        for v in (f.flops, f.bytes_loaded, f.bytes_stored, f.block_size, f.num_regs, f.n_spills)
    )
    assert (f.num_regs, f.n_spills) == (128, 12)


def test_block_size_derived_from_block_dims(tmp_path: Path) -> None:
    # block_size (threads/block, used by Layer 2 occupancy) is the product of block dims
    # when the caller doesn't give it explicitly.
    c = _make(tmp_path)
    c.add_kernel("k0", "k", compiled_kernel=_fake_compiled_kernel(), block=[64, 2, 1])
    assert c.kernels[0].features is not None
    assert c.kernels[0].features.block_size == 128


def test_explicit_kwarg_overrides_extracted(tmp_path: Path) -> None:
    # A caller-given value wins over what the kernel object reports (explicit beats inferred).
    c = _make(tmp_path)
    c.add_kernel("k0", "k", compiled_kernel=_fake_compiled_kernel(n_regs=40), num_regs=99)
    assert c.kernels[0].features is not None
    assert c.kernels[0].features.num_regs == 99


def test_missing_kernel_metadata_degrades_to_none(tmp_path: Path) -> None:
    # A kernel object missing attributes must not crash; the fields just stay unset.
    bare = SimpleNamespace()  # no n_regs, no metadata, no asm
    c = _make(tmp_path)
    c.add_kernel("k0", "k", compiled_kernel=bare)
    k = c.kernels[0]
    # No features and no launch_config could be derived -> both omitted entirely.
    assert k.features is None
    assert k.launch_config is None


def test_no_compiled_kernel_is_allowed(tmp_path: Path) -> None:
    # Caller-only path: metadata known without a live kernel object (e.g. replayed from a trace).
    c = _make(tmp_path)
    c.add_kernel("k0", "k", num_regs=32, n_spills=0, flops=1.0e9)
    f = c.kernels[0].features
    assert f is not None and f.num_regs == 32 and f.flops == pytest.approx(1.0e9)


# ── redaction of the kernel source ───────────────────────────────────────────────────────
def test_kernel_source_omitted_by_default(tmp_path: Path) -> None:
    # AC: kernel source is omitted by default (opt-in only).
    c = _make(tmp_path)
    c.add_kernel("k0", "k", kernel_source="@triton.jit\ndef k(...): ...")
    assert c.kernels[0].kernel_source_excerpt is None


def test_kernel_source_refused_under_strict_even_when_opted_in(tmp_path: Path) -> None:
    # Fail-closed (D11): opting in does NOT override a strict policy.
    c = _make(tmp_path, redaction_policy=RedactionPolicy.DEFAULT_STRICT)
    c.add_kernel("k0", "k", kernel_source="secret source", include_source=True)
    assert c.kernels[0].kernel_source_excerpt is None


def test_kernel_source_recorded_when_opted_in_under_non_strict(tmp_path: Path) -> None:
    c = _make(tmp_path, redaction_policy=RedactionPolicy.INTERNAL)
    c.add_kernel("k0", "k", kernel_source="@triton.jit def k(): ...", include_source=True)
    assert c.kernels[0].kernel_source_excerpt == "@triton.jit def k(): ..."


# ── path normalization at finalize ───────────────────────────────────────────────────────
def test_paths_normalized_under_strict(tmp_path: Path) -> None:
    repo = Path.cwd().name
    c = _make(tmp_path)
    c.add_kernel(
        "k0",
        "k",
        compiled_kernel=_fake_compiled_kernel(),
        ptx_path=f"/home/alice/work/{repo}/cache/k0.ptx",
        source_path=f"/home/alice/work/{repo}/kernels/k0.py",
    )
    c.finalize()
    k = c.kernels[0]
    assert k.ptx_path == "{repo}/cache/k0.ptx"
    assert k.source_path == "{repo}/kernels/k0.py"


def test_sensitive_path_refused(tmp_path: Path) -> None:
    from compile_lens.security.redactor import SensitivePathError

    c = _make(tmp_path)
    c.add_kernel("k0", "k", ptx_path="/home/alice/.ssh/leak.ptx")
    with pytest.raises(SensitivePathError):
        c.finalize()


# ── end-to-end: schema-valid artifact ────────────────────────────────────────────────────
def test_finalize_writes_valid_cls_json(tmp_path: Path) -> None:
    c = _make(tmp_path)
    c.add_kernel(
        "k0",
        "triton_poi_fused_add_0",
        compiled_kernel=_fake_compiled_kernel(),
        grid=[128, 1, 1],
        block=[256, 1, 1],
        flops=1.2e10,
        bytes_loaded=4.0e7,
        bytes_stored=2.0e7,
    )
    written = c.finalize()

    assert written == c.output_path
    text = written.read_text()

    # 1. round-trips through the pydantic binding (same gate test_schema.py uses)
    artifact = from_json(text)
    assert len(artifact.kernels) == 1
    assert artifact.kernels[0].kernel_id == "k0"

    # 2. validates against the published JSON Schema
    jsonschema.validate(json.loads(text), _SCHEMA)


def test_finalize_is_deterministic(tmp_path: Path) -> None:
    def _run(p: Path) -> str:
        c = _make(p)
        c.add_kernel("k0", "k", compiled_kernel=_fake_compiled_kernel(), block=[128, 1, 1])
        return c.finalize().read_text()

    assert _run(tmp_path / "a") == _run(tmp_path / "b")
