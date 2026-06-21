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

import math
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

#: Schema version these models mirror.
SCHEMA_VERSION: str = "0.5.0"


def _decode_non_finite(value: Any) -> Any:
    """Read side: map the JSON string sentinels back to floats; pass numbers through. (pydantic also
    coerces these by default; doing it explicitly keeps the contract symmetric with the writer and
    with the Rust adapter, and lets any *other* string be rejected rather than silently coerced.)"""
    if isinstance(value, str):
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}.get(value, value)
    return value


def _encode_non_finite(value: float) -> float | str:
    """Write side: a non-finite float becomes an explicit string sentinel; a finite one stays a
    JSON number."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


#: A ``float`` that encodes non-finite values (NaN, ±∞) as the explicit JSON string sentinels
#: ``"NaN"`` / ``"Infinity"`` / ``"-Infinity"`` instead of the silent ``null`` both serializers
#: default to — so a NaN survives the Python→Rust round trip and stays distinct from "not computed".
#: Applied to every nullable float that holds a computed or measured value. See ADR-039.
NonFiniteFloat = Annotated[
    float,
    BeforeValidator(_decode_non_finite),
    PlainSerializer(_encode_non_finite, return_type=float | str, when_used="json"),
]


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
    wall_clock_ms: NonFiniteFloat | None = None


class FxNode(_Model):
    #: Graph-unique node id; other nodes reference it in their ``inputs`` to encode an edge.
    id: str
    #: The operation, e.g. ``aten.matmul`` or a ``call_function`` target.
    op_type: str
    #: Ordered ids of the upstream nodes this one consumes. Order is load-bearing:
    #: ``sub(a, b)`` and ``sub(b, a)`` differ only here, so operand-order regressions are
    #: detectable only because the sequence is preserved rather than treated as a set.
    inputs: list[str] = Field(default_factory=list)
    #: Node attributes (a constant's value, a dim parameter, …); a difference here makes the
    #: diff classify an otherwise-identical node as ``modified``.
    attrs: dict[str, Any] = Field(default_factory=dict)
    #: Output tensor shape, when a concrete one was captured from ``node.meta['val']``. Empty when
    #: unknown (meta absent, value not a single tensor, or a dynamic/symbolic shape). Tool 6's cost
    #: model reads these to size HBM traffic; structure-only consumers ignore them (ADR-038).
    out_shape: list[int] = Field(default_factory=list)
    #: Output dtype as a torch name without the ``torch.`` prefix (``bfloat16``); absent under the
    #: same conditions as ``out_shape``.
    out_dtype: str | None = None


class CompiledGraph(_Model):
    graph_id: str
    compiled_function_id: str | None = None
    fx_graph_path: str | None = None
    inductor_ir_path: str | None = None
    guard_list: list[str] = Field(default_factory=list)
    kernel_ids: list[str] = Field(default_factory=list)
    compile_phases_summary: dict[str, Any] = Field(default_factory=dict)
    #: Node-level FX graph structure for the WL-signature diff (Tool 2a, ADR-024). Inlined so
    #: a single ``.cls.json`` stays self-contained; optional and forward-compatible.
    nodes: list[FxNode] = Field(default_factory=list)


class CompilePhase(_Model):
    name: str
    duration_ms: NonFiniteFloat | None = None


class Iteration(_Model):
    iteration_index: int
    timestamp_ms: NonFiniteFloat | None = None
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


# ── divergence (Tool 3; mirror records.rs Divergence / DivergenceAttribution, ADR-034) ───
class DivergenceAttribution(_Model):
    """Pass-level causal attribution for a divergence (ADR-032 / ADR-034). Pass-level only:
    ``responsible_passes`` names the inductor pass(es) whose disabling removes the divergence,
    never a fused node."""

    attributed: bool
    responsible_passes: list[str] = Field(default_factory=list)
    summary: str
    num_probes: int


class Divergence(_Model):
    """One eager-vs-compiled divergence finding (``divergences[]``). An *analysis result*, so a
    normalized array record like ``lint_findings`` (ADR-021/034); the attribution is nested.
    ``first_divergent_layer`` is absent when nothing diverged; ``max_abs_diff`` is absent for a
    shape mismatch or no divergence."""

    divergence_id: str
    first_divergent_layer: str | None = None
    max_abs_diff: NonFiniteFloat | None = None
    num_layers_compared: int
    rtol: float
    atol: float
    suggested_cause: str | None = None
    attribution: DivergenceAttribution | None = None


# ── kernels + roofline (mirror kernels.rs) ──────────────────────────────────────────────
class LaunchConfig(_Model):
    grid: list[int] = Field(default_factory=list)
    block: list[int] = Field(default_factory=list)
    shared_mem_bytes: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None


class KernelFeatures(_Model):
    flops: NonFiniteFloat | None = None  # schema `number` -> float (arrives as 1.2e10 etc.)
    bytes_loaded: NonFiniteFloat | None = None
    bytes_stored: NonFiniteFloat | None = None
    block_size: int | None = None
    num_regs: int | None = None  # register pressure proxy (Tool 5 4th correction)
    n_spills: int | None = None


class KernelMeasurements(_Model):
    iterations: int | None = None
    mean_us: NonFiniteFloat | None = None
    median_us: NonFiniteFloat | None = None
    p99_us: NonFiniteFloat | None = None
    source: str | None = None  # plain str ("proton"/"self_timed"); forward-compat


class Kernel(_Model):
    kernel_id: str
    name: str
    source_path: str | None = None
    ptx_path: str | None = None
    kernel_source_excerpt: str | None = None  # opt-in only
    launch_config: LaunchConfig | None = None
    occupancy_hint: NonFiniteFloat | None = None
    features: KernelFeatures | None = None
    measurements: KernelMeasurements | None = None
    autotune_history: list[dict[str, Any]] = Field(default_factory=list)


class GpuSpec(_Model):
    name: str | None = None
    peak_compute_tflops_bf16: NonFiniteFloat | None = None
    peak_bw_gbps: NonFiniteFloat | None = None
    l2_cache_mb: NonFiniteFloat | None = None
    sm_count: int | None = None


class Calibration(_Model):
    pearson_correlation: NonFiniteFloat | None = None  # float (may be negative)
    spearman_correlation: NonFiniteFloat | None = None
    calibration_verdict: str | None = None
    n_samples: int | None = None


class PruningDecision(_Model):
    mode: str | None = None  # plain str; forward-compat
    threshold_rank_correlation: NonFiniteFloat | None = None
    configs_total: int | None = None
    configs_predicted_top_k: int | None = None
    configs_measured: int | None = None


class Suggestion(_Model):
    text: str | None = None
    evidence_link: str | None = None


class RooflinePrediction(_Model):
    kernel_id: str
    gpu_spec: GpuSpec | None = None
    arithmetic_intensity: NonFiniteFloat | None = None
    bound_type: str | None = None  # plain str ("memory"/"compute"); forward-compat
    theoretical_lower_bound_us: NonFiniteFloat | None = None
    empirical_predictor_us: NonFiniteFloat | None = None
    measured_us: NonFiniteFloat | None = None
    achieved_vs_predictor: NonFiniteFloat | None = None
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


class FusionShape(_Model):
    m: int | None = None
    n: int | None = None
    k0: int | None = None
    k1: int | None = None
    dtype: str | None = None


class FusionLocation(_Model):
    fx_node_ids: list[str] = Field(default_factory=list)
    src_lineno_range: str | None = None


class FusionOpportunity(_Model):
    """Tool 6 algebraic fusion opportunity (``fusion_opportunities[]``). Suggest-only — the
    detector identifies and quantifies the opportunity and describes the fusion to apply in
    ``suggested_fusion``; it never writes, imports, or runs a kernel."""

    pattern_id: str
    location: FusionLocation | None = None
    shape: FusionShape | None = None
    baseline_hbm_bytes: NonFiniteFloat | None = None
    fused_hbm_bytes: NonFiniteFloat | None = None
    estimated_speedup: NonFiniteFloat | None = None
    suggested_fusion: str | None = None
    confidence: str | None = None  # "high" / "medium" / "low" (plain str, forward-compat)


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
    divergences: list[Divergence] = Field(default_factory=list)
    kernels: list[Kernel] = Field(default_factory=list)
    roofline_predictions: list[RooflinePrediction] = Field(default_factory=list)
    fusion_opportunities: list[FusionOpportunity] = Field(default_factory=list)


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
