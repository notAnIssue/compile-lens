# ADR-021: Schema v0.5.0 top-level layout

- **Status**: Accepted
- **Date**: 2026-05-26
- **Deciders**: project maintainer
- **Related**: `schema/v0.5.0.json` (this decision's implementation). Unblocks the Rust (serde) and Python (Pydantic) schema bindings and the cross-language round-trip test.

## Context

The `.cls.json` artifact is the **sole interface between the Python collectors and the
Rust analyzers** — there is no PyO3 and no IPC, only a file on disk. Its shape *is* the
API. Once locked at `v0.5.0` under SemVer, changing it costs a migration:

- **Adding a field** is cheap (add field + default; near-identity migration).
- **Changing a field's type/meaning** is moderate (one migration function).
- **Reshaping (flat ↔ nested, moving a field between groups)** is the expensive case:
  every historical artifact must be transformed and the N-1 compatibility window must be
  re-tested.

The collector follows a generosity principle — it records signals even when no current
analyzer consumes them — so the field set will only grow. The decision must therefore be
made for a schema that gets *larger*, not for today's ~17 fields. **The goal of this
decision is to minimize the probability of a future *reshape* migration**, not to freeze
the field set.

The `session` object's question is *not* "flat vs nested": two fields
(`compile_config.options`, `env_snapshot.relevant_env_vars`) are open-ended maps that
cannot be flattened, so some nesting is mandatory. The real question is **which attribute
of a field decides the group it belongs to** (the "grouping axis"), since a field has
several attributes (semantic domain, sensitivity, reproducibility-relevance) but can
physically live in only one place.

**Pseudo-criterion, explicitly rejected:** an earlier framing listed "query performance"
as a trade-off axis. There is no query engine — the artifact is deserialized whole into
in-memory structs, and field-access depth (`session.compile_config.backend` vs
`session.backend`) is a compile-time concern with negligible runtime cost. The real axes
are *ergonomics*, *migration reshape risk*, *namespace*, and *redaction implementability*.

## Decision

Adopt a **semantic-domain hybrid** layout, governed by an explicit, reusable rule:

> **"Nest objects, flatten scalars."**
> 1. A field that is itself an object or map (has internal structure) → its own nested
>    group (`compile_config`, `env_snapshot`, `collector_versions`).
> 2. A field that is an atomic scalar (string / number / enum) → stays flat on `session`.
> 3. The grouping axis is **semantic domain** (where the data comes from / how it is read
>    as a unit) — **not** sensitivity, **not** reproducibility.
> 4. **Sensitivity is a field-level redaction map** consulted by `cls-scrub`, never a
>    structural axis — so reclassifying a field's sensitivity never reshapes the schema.
> 5. The top-level artifact is **normalized**: `session` plus parallel arrays
>    (`graph_breaks[]`, `recompilations[]`, `compiled_graphs[]`, `kernels[]`, …) that
>    cross-reference by `*_id`, rather than embedding records inside one another.

The resulting `session` layout:

```jsonc
"session": {
  // flat scalars
  "id", "timestamp",
  "torch_version", "triton_version", "cuda_version", "python_version",
  "gpu_name", "host", "command", "duration_ms", "iteration_count",
  "git_sha", "workspace_fingerprint", "rank", "world_size", "redaction_policy",
  // nested objects (internal structure / open maps)
  "collector_versions": { ... },
  "compile_config":    { "fullgraph", "dynamic", "backend", "mode", "options": {} },
  "env_snapshot":      { "relevant_env_vars": {}, "random_seed" }
}
```

Version fields and `rank`/`world_size` stay flat (they are atomic scalars and are
frequently accessed standalone — by-commit comparison, by-rank filtering); they are *not*
grouped into `versions`/`distributed` objects, per rule (2).

## Consequences

- **Positive**:
  - The grouping axis (semantic domain) is the most *stable* one: data sources
    (`torch.compile()` call, `os.environ`, runtime introspection) do not change, so the
    probability of a forced reshape migration is minimized.
  - Open maps are represented naturally; nested objects map cleanly onto serde structs,
    Pydantic models, and reusable JSON Schema `$defs`.
  - The layout matches both the producer (collectors write by source) and the consumer
    (analyzers read a domain as a unit, e.g. the recompile summary reads `compile_config`
    whole).
- **Negative / costs**:
  - Sensitivity handling must be implemented as a separate field-level redaction map in
    `cls-scrub` (cannot "clear one subtree"). Accepted cost — see Alternatives, Option C.
  - JSON imposes a determinism discipline: pin float formatting and use an order-preserving
    map for deterministic key ordering (identical inputs must serialize byte-for-byte);
    encode `NaN`/`Inf` explicitly (JSON has no native representation) for divergence values.
- **Follow-ups**:
  - `cls-scrub` owns the field-level redaction map keyed by `redaction_policy`.
  - The reproducibility bundle (the inputs needed to replay a session: `compile_config` +
    `env_snapshot` + `git_sha` + `random_seed` + `rank` + `world_size`) is expressed as a
    collector-side helper, not a structural group.

## Alternatives considered

- **Option A — fully flat**: all fields at one level. Near-infeasible: the two open maps
  cannot become named struct fields, so "flat" only removes the semantic wrappers without
  removing nesting, while losing grouping.
- **Option B — semantic-domain hybrid** (chosen): nest by what the data *is*.
- **Option C — sensitivity hybrid**: group all redactable fields into a `sensitive`
  subtree. Tempting because redaction becomes "clear one subtree," but it couples layout
  to security policy: sensitivity gets *reclassified* (e.g. `gpu_name` public → confidential),
  and every reclassification then forces a reshape migration — the most expensive case.

### Weighted decision matrix

Criteria weighted to sum to 10; reasoning for each weight given. Every option scored
0–10 on the same scale; weighted contribution = `weight × (raw/10)`.

| Criterion (weight) — why this weight | A flat | B semantic ✅ | C sensitivity |
|---|---|---|---|
| **Resistance to future reshape migration (3.0)** — the single most expensive failure mode; this is what the decision exists to minimize | 4 (1.20) | **9 (2.70)** | 2 (0.60) |
| **Technical feasibility (2.0)** — near-hard-constraint: open maps cannot be flattened | 2 (0.40) | **10 (2.00)** | 7 (1.40) |
| **Redaction decoupled from layout (1.5)** — security-policy changes must not reshape the schema | 5 (0.75) | **9 (1.35)** | 6 (0.90) |
| **Producer + consumer ergonomics (1.5)** — write-by-source, read-by-domain | 4 (0.60) | **9 (1.35)** | 4 (0.60) |
| **Type-system / `$defs` fit (1.0)** — serde / Pydantic / JSON Schema reuse | 3 (0.30) | **9 (0.90)** | 5 (0.50) |
| **Teachability / low churn (1.0)** — a crisp rule future fields obey | 5 (0.50) | **9 (0.90)** | 4 (0.40) |
| **Weighted total / 10** | **3.75** | **9.20** | **4.40** |

**Readout**: B wins decisively (9.20 vs C 4.40 vs A 3.75), dominating on the two
highest-weighted criteria (reshape resistance + feasibility = 50% of total weight). C's
"clear one subtree" benefit is outweighed by its bottom score on reshape resistance —
confirming that rejecting the sensitivity axis is the load-bearing part of this decision.
The result would only soften if "reshape resistance" were significantly down-weighted
(i.e. if the schema were not expected to outlive a couple of minor versions).
