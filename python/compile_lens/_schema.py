"""Pydantic v2 models for the compile-lens ``.cls.json`` session artifact (schema v0.5.0).

This is the **Python half** of the cross-language contract; the Rust half is the
``cls-schema`` crate. The two must produce/consume identical JSON so the cross-language
round-trip test (Python writes -> Rust reads -> consistent, and reverse) passes.
These models mirror ``schema/v0.5.0.json`` and ``crates/cls-schema/src/*.rs`` 1:1; the
per-decision rationale lives in the comments below.

Representation conventions (mirroring the Rust side):

- **Numbers**: schema ``integer`` -> ``int``, ``number`` -> ``float`` (incl. ``flops`` /
  ``bytes_*``, which arrive in scientific notation), ``string`` -> ``str``.
- **Unknown-field capture (DR4)**: only :class:`ClsArtifact` and :class:`Session` use
  ``extra="allow"`` (unknown keys preserved through a round trip). Every other model uses
  ``extra="ignore"`` (unknown keys accepted on read, dropped on write).
- **Optional vs. nullable (DR3)**: serialization uses ``exclude_unset=True``
  (see :func:`to_json`), so a key absent from the input stays absent and a present ``null``
  stays ``null``. The two schema ``[T, null]`` union fields
  (:attr:`CompileConfig.dynamic`, :attr:`EnvSnapshot.random_seed`) are meaningful when
  ``null``; a producer constructing a model from scratch must pass them explicitly so they
  are emitted (see their field docstrings).
- **Fail-closed enum (DR6)**: :class:`RedactionPolicy` is the only constrained string
  modeled as an enum; an unknown value raises ``ValidationError``. Other constrained
  strings stay ``str`` for forward compatibility (D6).
- **Determinism (DR5)**: Python ``dict`` preserves insertion order, so open maps need no
  special handling (the Rust side uses ``IndexMap`` for the same reason). Never serialize
  with ``sort_keys``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Schema version these models mirror.
SCHEMA_VERSION: str = "0.5.0"


class _Model(BaseModel):
    """Shared base: unknown keys are accepted but not preserved (``extra="ignore"``).

    The two growth-point models override this with ``extra="allow"``.
    """

    model_config = ConfigDict(extra="ignore")


class RedactionPolicy(StrEnum):
    """Redaction classification (D11). Closed + fail-closed: an unknown wire value raises
    ``ValidationError`` rather than silently defaulting to a more permissive policy."""

    DEFAULT_STRICT = "default-strict"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PUBLIC_SAFE = "public-safe"


# ── shared location types ───────────────────────────────────────────────────────────────
class SourceLocation(_Model):
    file: str | None = None
    line: int | None = None
    function: str | None = None


class SourceRange(_Model):
    file: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    code_excerpt: str | None = None


# ── records (mirror records.rs) ─────────────────────────────────────────────────────────
class FailedGuard(_Model):
    guard_id: str | None = None
    expression: str | None = None
    previous_value: str | None = None
    new_value: str | None = None


class GuardEvaluation(_Model):
    guard_id: str | None = None
    evaluation_result: str | None = None
    evaluated_value: str | None = None


class InternalStateSnapshot(_Model):
    module_buffers_hash: str | None = None
    module_attrs_changed: list[str] = Field(default_factory=list)


class ReferenceIssue(_Model):
    url: str | None = None
    fixed_in_version: str | None = None


class GraphBreak(_Model):
    break_id: str
    reason: str
    location: SourceLocation | None = None
    op_or_construct: str | None = None
    fallback_kind: str | None = None


class Recompilation(_Model):
    recompilation_id: str
    compiled_function_id: str | None = None
    trigger_reason: str | None = None
    failed_guard: FailedGuard | None = None
    occurred_at_step: int | None = None
    wall_clock_ms: float | None = None


class CompiledGraph(_Model):
    graph_id: str
    compiled_function_id: str | None = None
    fx_graph_path: str | None = None
    inductor_ir_path: str | None = None
    guard_list: list[str] = Field(default_factory=list)
    kernel_ids: list[str] = Field(default_factory=list)
    compile_phases_summary: dict[str, Any] = Field(default_factory=dict)


class CompilePhase(_Model):
    name: str
    duration_ms: float | None = None


class Iteration(_Model):
    iteration_index: int
    timestamp_ms: float | None = None
    guard_evaluations: list[GuardEvaluation] = Field(default_factory=list)
    cache_hit: bool | None = None
    recompilation_triggered: bool | None = None
    output_signature: str | None = None
    internal_state_snapshot: InternalStateSnapshot | None = None


class LintFinding(_Model):
    finding_id: str
    pattern_category: str  # plain str (vocabulary grows post-v1.0); not an enum (D6)
    severity: str
    source_location: SourceRange | None = None
    trigger_pattern: str | None = None
    li_et_al_section: str | None = None
    reference_issue: ReferenceIssue | None = None
    workaround: str | None = None
    confidence: str | None = None
    applies_to_user_torch_version: bool | None = None


# ── kernels + roofline (mirror kernels.rs) ──────────────────────────────────────────────
class LaunchConfig(_Model):
    grid: list[int] = Field(default_factory=list)
    block: list[int] = Field(default_factory=list)
    shared_mem_bytes: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None


class KernelFeatures(_Model):
    flops: float | None = None  # schema `number` -> float (arrives as 1.2e10 etc.)
    bytes_loaded: float | None = None
    bytes_stored: float | None = None
    block_size: int | None = None
    num_regs: int | None = None  # register pressure proxy (Tool 5 4th correction)
    n_spills: int | None = None


class KernelMeasurements(_Model):
    iterations: int | None = None
    mean_us: float | None = None
    median_us: float | None = None
    p99_us: float | None = None
    source: str | None = None  # plain str ("proton"/"self_timed"); forward-compat


class Kernel(_Model):
    kernel_id: str
    name: str
    source_path: str | None = None
    ptx_path: str | None = None
    kernel_source_excerpt: str | None = None  # opt-in only
    launch_config: LaunchConfig | None = None
    occupancy_hint: float | None = None
    features: KernelFeatures | None = None
    measurements: KernelMeasurements | None = None
    autotune_history: list[dict[str, Any]] = Field(default_factory=list)


class GpuSpec(_Model):
    name: str | None = None
    peak_compute_tflops_bf16: float | None = None
    peak_bw_gbps: float | None = None
    l2_cache_mb: float | None = None
    sm_count: int | None = None


class Calibration(_Model):
    pearson_correlation: float | None = None  # float (may be negative)
    spearman_correlation: float | None = None
    calibration_verdict: str | None = None
    n_samples: int | None = None


class PruningDecision(_Model):
    mode: str | None = None  # plain str; forward-compat
    threshold_pearson: float | None = None
    configs_total: int | None = None
    configs_predicted_top_k: int | None = None
    configs_measured: int | None = None


class Suggestion(_Model):
    text: str | None = None
    evidence_link: str | None = None


class RooflinePrediction(_Model):
    kernel_id: str
    gpu_spec: GpuSpec | None = None
    arithmetic_intensity: float | None = None
    bound_type: str | None = None  # plain str ("memory"/"compute"); forward-compat
    theoretical_lower_bound_us: float | None = None
    empirical_predictor_us: float | None = None
    measured_us: float | None = None
    achieved_vs_predictor: float | None = None
    corrections_applied: list[str] = Field(default_factory=list)
    calibration: Calibration | None = None
    pruning_decision: PruningDecision | None = None
    verdict: str | None = None
    suggestions: list[Suggestion] = Field(default_factory=list)


# ── session (mirror session.rs) ─────────────────────────────────────────────────────────
class CompileConfig(_Model):
    fullgraph: bool | None = None
    #: ``torch.compile(dynamic=...)``. Schema type ``[boolean, null]``: ``null`` means
    #: "auto" and is a distinct, meaningful state. On a round trip (parsed input) a present
    #: ``null`` is preserved by ``exclude_unset``. A producer building this from scratch
    #: must pass ``dynamic=`` explicitly (even ``None``) for it to be emitted.
    dynamic: bool | None = None
    backend: str | None = None
    mode: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)  # open map; cannot be flattened


class EnvSnapshot(_Model):
    relevant_env_vars: dict[str, str] = Field(default_factory=dict)
    #: Schema type ``[integer, null]``: a present-``null`` seed (deliberately unset) differs
    #: from "no seed field". Same construct-vs-parse caveat as :attr:`CompileConfig.dynamic`.
    random_seed: int | None = None


class Session(_Model):
    """Run-level metadata. Growth point: ``extra="allow"`` preserves unknown session keys
    (e.g. a future ``gpu_arch``) through a round trip, mirroring the Rust ``Session.extra``."""

    model_config = ConfigDict(extra="allow")

    id: str
    timestamp: str
    torch_version: str
    redaction_policy: RedactionPolicy
    triton_version: str | None = None
    cuda_version: str | None = None
    python_version: str | None = None
    gpu_name: str | None = None
    host: str | None = None
    command: str | None = None
    duration_ms: int | None = None
    iteration_count: int | None = None
    git_sha: str | None = None
    workspace_fingerprint: str | None = None
    rank: int | None = None
    world_size: int | None = None
    collector_versions: dict[str, str] = Field(default_factory=dict)
    compile_config: CompileConfig | None = None
    env_snapshot: EnvSnapshot | None = None


class ClsArtifact(_Model):
    """Top-level ``.cls.json`` artifact. Growth point: ``extra="allow"`` preserves unknown
    top-level keys (e.g. a future record array), mirroring the Rust ``ClsArtifact.extra``."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    session: Session
    graph_breaks: list[GraphBreak] = Field(default_factory=list)
    recompilations: list[Recompilation] = Field(default_factory=list)
    compiled_graphs: list[CompiledGraph] = Field(default_factory=list)
    compile_phases: list[CompilePhase] = Field(default_factory=list)
    iterations: list[Iteration] = Field(default_factory=list)
    lint_findings: list[LintFinding] = Field(default_factory=list)
    kernels: list[Kernel] = Field(default_factory=list)
    roofline_predictions: list[RooflinePrediction] = Field(default_factory=list)


def from_json(data: str | bytes) -> ClsArtifact:
    """Parse a ``.cls.json`` document into a :class:`ClsArtifact`."""
    return ClsArtifact.model_validate_json(data)


def to_json(artifact: ClsArtifact) -> str:
    """Serialize a :class:`ClsArtifact` back to JSON.

    Uses ``exclude_unset=True`` so the output mirrors serde's behavior: keys that were
    absent in the parsed input stay absent, while a present ``null`` (e.g. ``dynamic``)
    is preserved. This is the faithful round-trip path the Rust side reads back.
    """
    return artifact.model_dump_json(exclude_unset=True)
