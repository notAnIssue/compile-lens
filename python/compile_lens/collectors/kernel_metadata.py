"""``KernelMetadataCollector`` — capture Triton kernel metadata into a ``.cls.json`` (Tool 5).

Tool 5 (kernel roofline triage) needs, per generated Triton kernel, the inputs its cost model
reads: the launch configuration, the kernel *features* (FLOPs, bytes moved, register pressure),
and a pointer to the PTX. This collector assembles those into the ``kernels[]`` contract.

Responsibilities are split by where the fact actually lives:

  * the **compiled-kernel object** (a duck-typed Triton ``CompiledKernel``) carries the
    compile-product facts — ``n_regs`` / ``n_spills`` and, on its ``.metadata``, ``num_warps`` /
    ``num_stages`` / shared-memory bytes. These are read with guarded ``getattr`` so a missing
    or version-shifted attribute degrades to ``None`` instead of raising;
  * the **caller** supplies the launch + semantic facts that are *not* in a kernel's static
    metadata — the launch ``grid`` / ``block`` and the algorithmic ``flops`` / ``bytes_loaded`` /
    ``bytes_stored`` (these come from the op the kernel implements, not from the kernel binary).

Redaction (discipline D11): ``kernel_source_excerpt`` is opt-in and, even when opted in, is
refused under a strict policy; ``ptx_path`` / ``source_path`` are normalized at ``finalize``.

``torch`` is an optional dependency, imported lazily only to stamp the session's torch version.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from compile_lens._schema import (
    SCHEMA_VERSION,
    ClsArtifact,
    Kernel,
    KernelFeatures,
    LaunchConfig,
    RedactionPolicy,
    Session,
    to_json,
)

_log = structlog.get_logger("compile_lens.collectors.kernel_metadata")


# ── duck-typed extraction off a Triton CompiledKernel ─────────────────────────────────────
def _meta_attr(metadata: Any, name: str) -> Any:
    """Read ``name`` off a kernel's ``.metadata``, which is an object on some Triton versions
    and a mapping on others. Returns ``None`` if absent."""
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _extract_compiled_kernel(compiled_kernel: Any) -> dict[str, Any]:
    """Duck-type the compile-product facts off a Triton ``CompiledKernel``.

    Triton exposes ``n_regs`` / ``n_spills`` as top-level attributes and ``num_warps`` /
    ``num_stages`` / ``shared`` (shared-memory bytes) on ``.metadata``. These names are
    version-sensitive, so each read is a guarded ``getattr`` — a missing attribute becomes
    ``None`` rather than an error (mirrors how the artifact collector duck-types FX nodes).
    """
    keys = ("num_regs", "n_spills", "num_warps", "num_stages", "shared_mem_bytes")
    if compiled_kernel is None:
        return dict.fromkeys(keys)
    metadata = getattr(compiled_kernel, "metadata", None)
    return {
        "num_regs": getattr(compiled_kernel, "n_regs", None),
        "n_spills": getattr(compiled_kernel, "n_spills", None),
        "num_warps": _meta_attr(metadata, "num_warps"),
        "num_stages": _meta_attr(metadata, "num_stages"),
        "shared_mem_bytes": _meta_attr(metadata, "shared"),
    }


def _pick(explicit: Any, extracted: Any) -> Any:
    """The caller's explicit value wins; otherwise fall back to what was extracted."""
    return explicit if explicit is not None else extracted


def _build_features(
    *,
    flops: float | None,
    bytes_loaded: float | None,
    bytes_stored: float | None,
    block_size: int | None,
    num_regs: int | None,
    n_spills: int | None,
) -> KernelFeatures | None:
    """A ``KernelFeatures`` if any field is set, else ``None`` (so an empty struct is omitted)."""
    fields: dict[str, Any] = {
        "flops": flops,
        "bytes_loaded": bytes_loaded,
        "bytes_stored": bytes_stored,
        "block_size": block_size,
        "num_regs": num_regs,
        "n_spills": n_spills,
    }
    present = {k: v for k, v in fields.items() if v is not None}
    return KernelFeatures(**present) if present else None


def _build_launch_config(
    *,
    grid: list[int] | None,
    block: list[int] | None,
    shared_mem_bytes: int | None,
    num_warps: int | None,
    num_stages: int | None,
) -> LaunchConfig | None:
    """A ``LaunchConfig`` if any field is set, else ``None``."""
    fields: dict[str, Any] = {}
    if grid:
        fields["grid"] = list(grid)
    if block:
        fields["block"] = list(block)
    if shared_mem_bytes is not None:
        fields["shared_mem_bytes"] = shared_mem_bytes
    if num_warps is not None:
        fields["num_warps"] = num_warps
    if num_stages is not None:
        fields["num_stages"] = num_stages
    return LaunchConfig(**fields) if fields else None


class KernelMetadataCollector:
    """Captures per-kernel metadata and writes a schema-valid ``.cls.json``.

    Mirrors :class:`compile_lens.collectors.artifact.CompileArtifactCollector`'s session +
    redaction handling. The capture surface is :meth:`add_kernel`; :meth:`finalize` writes the
    artifact.

    Args:
        output_path: where :meth:`finalize` writes the ``.cls.json``.
        redaction_policy: classification for the session (D11), coerced through
            :class:`RedactionPolicy` so an unknown value fails closed.
        session_id / timestamp / torch_version: session metadata; each defaults to a runtime
            value when ``None`` (fresh UUID4, current UTC time, installed torch version), and
            tests pass them explicitly for deterministic output.
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
        session_id: str | None = None,
        timestamp: str | None = None,
        torch_version: str | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.redaction_policy = RedactionPolicy(redaction_policy)  # fail-closed (D11)
        self._session_id = session_id
        self._timestamp = timestamp
        self._torch_version = torch_version
        self.kernels: list[Kernel] = []

    def add_kernel(
        self,
        kernel_id: str,
        name: str,
        *,
        compiled_kernel: Any = None,
        grid: list[int] | None = None,
        block: list[int] | None = None,
        flops: float | None = None,
        bytes_loaded: float | None = None,
        bytes_stored: float | None = None,
        block_size: int | None = None,
        num_regs: int | None = None,
        n_spills: int | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
        shared_mem_bytes: int | None = None,
        ptx_path: str | None = None,
        source_path: str | None = None,
        kernel_source: str | None = None,
        include_source: bool = False,
    ) -> None:
        """Assemble one ``kernels[]`` record.

        Values read off ``compiled_kernel`` (``num_regs`` / ``n_spills`` / ``num_warps`` /
        ``num_stages`` / shared memory) are overridden by any explicit keyword. ``block_size``
        (threads per block, read by the Layer-2 occupancy correction) is derived as the product
        of ``block`` when not given. ``kernel_source`` is recorded only when ``include_source``
        is set *and* the policy is not strict — opting in never overrides a strict policy.
        """
        from compile_lens.security import redactor  # noqa: PLC0415

        extracted = _extract_compiled_kernel(compiled_kernel)
        num_regs = _pick(num_regs, extracted["num_regs"])
        n_spills = _pick(n_spills, extracted["n_spills"])
        num_warps = _pick(num_warps, extracted["num_warps"])
        num_stages = _pick(num_stages, extracted["num_stages"])
        shared_mem_bytes = _pick(shared_mem_bytes, extracted["shared_mem_bytes"])

        if block_size is None and block:
            block_size = math.prod(block)

        features = _build_features(
            flops=flops,
            bytes_loaded=bytes_loaded,
            bytes_stored=bytes_stored,
            block_size=block_size,
            num_regs=num_regs,
            n_spills=n_spills,
        )
        launch_config = _build_launch_config(
            grid=grid,
            block=block,
            shared_mem_bytes=shared_mem_bytes,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        record_source = include_source and not redactor.is_strict(self.redaction_policy)
        excerpt = kernel_source if record_source else None

        fields: dict[str, Any] = {"kernel_id": kernel_id, "name": name}
        if ptx_path is not None:
            fields["ptx_path"] = ptx_path
        if source_path is not None:
            fields["source_path"] = source_path
        if excerpt is not None:
            fields["kernel_source_excerpt"] = excerpt
        if launch_config is not None:
            fields["launch_config"] = launch_config
        if features is not None:
            fields["features"] = features
        self.kernels.append(Kernel(**fields))

    # ── serialization ──────────────────────────────────────────────────────────────────
    def _build_session(self) -> Session:
        def _detect_torch_version() -> str:
            try:
                import torch  # noqa: PLC0415

                return str(torch.__version__)
            except Exception:  # pragma: no cover — environment-dependent
                return "unknown"

        return Session(
            id=self._session_id or str(uuid.uuid4()),
            timestamp=self._timestamp or datetime.now(UTC).isoformat(),
            torch_version=self._torch_version or _detect_torch_version(),
            redaction_policy=self.redaction_policy,
        )

    def finalize(self) -> Path:
        """Write the captured kernels to a schema-valid ``.cls.json``.

        Under a strict policy, ``ptx_path`` / ``source_path`` are normalized at write time (D11);
        a path inside a credential directory raises
        :class:`~compile_lens.security.redactor.SensitivePathError`.
        """
        from compile_lens.security import redactor  # noqa: PLC0415

        if redactor.is_strict(self.redaction_policy):
            repo = Path.cwd().name
            for kernel in self.kernels:
                if kernel.ptx_path is not None:
                    kernel.ptx_path = redactor.normalize_path(kernel.ptx_path, repo=repo)
                if kernel.source_path is not None:
                    kernel.source_path = redactor.normalize_path(kernel.source_path, repo=repo)

        artifact = ClsArtifact(
            schema_version=SCHEMA_VERSION,
            session=self._build_session(),
            kernels=self.kernels,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(to_json(artifact))
        _log.debug(
            "kernel_metadata_collector.finalize",
            output_path=str(self.output_path),
            kernels=len(self.kernels),
            redaction_policy=str(self.redaction_policy),
        )
        return self.output_path
