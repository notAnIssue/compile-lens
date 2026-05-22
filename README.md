# compile-lens

<!-- demo-gif-placeholder: a 10-15s screencast lands here at v0.5.0 ship (see S7.X). -->

> **compile-lens — a diagnostic suite for `torch.compile` production observability (Python + Rust), turning PyTorch's official tracing evidence into workflow-ready insights for recompile/regression/divergence/lint/kernel triage.**

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](./LICENSE)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#roadmap)

---

## Roadmap

| Release | Status | Headline deliverable |
|---|---|---|
| **v0.5.0** (MVP) | **WIP** | Tool 1 (`recompile-summary`) + Tool 2a (`compile-diff`) + Hero `cl.session()` end-to-end |
| **v0.6.0** (Feature complete) | Planned | Full 5-tool toolkit: Tools 2b / 3 / 4 / 5 |
| **v0.7.0** (Agentic) | Planned | Sandboxed MCP server for LLM-agent consumption |
| **v1.0.0** (Stable) | TBD | Public API freeze + the seven readiness criteria |

See [`docs/01_roadmap.md`](./docs/01_roadmap.md) for the per-phase breakdown (post-Phase 0).

---

## Why this exists

PyTorch already ships excellent **raw-evidence** tooling for `torch.compile`:

- [`tlparse`](https://github.com/pytorch/tlparse) — `TORCH_TRACE` log parser
- `TORCH_LOGS=+recompiles / +graph_breaks / +inductor` — built-in structured logs
- [`depyf`](https://github.com/thuml/depyf) — Dynamo bytecode decompiler
- [`proton`](https://github.com/triton-lang/triton/tree/main/third_party/proton) — Triton's official kernel profiler

What is missing is the **interpretation / CI-gate / team-workflow** layer on top of that evidence:

1. **Structured interpretation** — guard-failure clustering, IR-level workload-signature diffing, roofline overlay on kernel profiles.
2. **CI gating** — turning a "torch.compile got slower in this PR" hunch into a deterministic SARIF finding a reviewer can block on.
3. **Team workflow** — a single `cl.session()` context that captures everything a reviewer or on-call needs, exportable as one HTML report.

compile-lens wraps the official tools (it does not rewrite them) and fills those three gaps. It is **not** a replacement for `torch.compile`, a fuzzer, a correctness oracle, or a competing compiler — see [§1.4 of the design doc](./docs/00_design/design.md#14-not-this) for the explicit non-goals.

---

## Background

compile-lens grew out of the author's own experience porting and adapting models to `torch.compile` across different hardware targets. That work surfaced the same tedious loop again and again: chase down a recompilation storm by hand, scroll thousands of `TORCH_LOGS` lines to find the one guard that's flapping, manually diff two compile dumps to figure out why a PR regressed, bisect an eager-vs-compile divergence with ad-hoc hooks. Each task was solvable, but each one cost hours and the tooling lived as throwaway scripts in private notebooks. compile-lens is the distilled version of those scripts — turned into a reproducible, schema-backed toolkit so that the next person hitting the same wall does not have to start from zero.

The framing is grounded in *Li et al. 2026* on `torch.compile` bug taxonomy: `torch.compile` is the **most-reported high-priority PyTorch component (46.6 % of high-priority bugs)**, and **correctness bugs alone account for 41.3 % of those** — yet no unified diagnostic workflow exists. compile-lens aims to close that gap as a polite ecosystem citizen on top of the official PyTorch tooling, not as a parallel stack.

---

## Install

> **Pre-alpha — installation will be wired up in S0.13.** The commands below are the target shape; they are not expected to work yet.

```bash
# From PyPI (planned)
pip install compile-lens

# From source (dev)
git clone https://github.com/<owner>/compile-lens.git
cd compile-lens
pip install -e '.[dev]'
```

Requirements (target): Python ≥ 3.11, PyTorch ≥ 2.5, Linux or macOS (CPU works; CUDA optional for kernel tooling).

---

## Quick example

> **Placeholder — the real worked example lands in Phase 7 (Hero form) with the demo gif.**

```python
import compile_lens as cl
import torch

model = torch.compile(MyModel())

with cl.session() as s:
    for batch in dataloader:
        model(batch)

s.report("triage.html")   # one HTML, every tool's findings
```

```bash
cl session report ./.compile-lens/<session-id>/   # same data, CLI form
```

---

## Documentation

- [Design doc](./docs/00_design/design.md) — architecture, design principles, ADR index
- [Roadmap](./docs/01_roadmap.md) — phase-by-phase plan
- [Tool pages](./docs/03_tools/) — one page per tool, with worked examples and limitations
- [Security policy](./SECURITY.md) — vulnerability reporting, disclosure timeline
- [Threat model](./docs/06_security/threat_model.md) — STRIDE analysis
- [Redaction policy](./docs/06_security/redaction_policy.md) — what gets captured, what gets scrubbed
- [Changelog](./CHANGELOG.md) — Keep-a-Changelog format
- [License](./LICENSE) — BSD-3-Clause (aligned with `tlparse`)

---

## Contributing

Pre-alpha — external contributions are not being accepted yet. The repository is public for transparency on design and progress. Once v0.5.0 ships, a `CONTRIBUTING.md` will describe the workflow.

If you have found a security issue, please follow [`SECURITY.md`](./SECURITY.md) — do **not** open a public issue.

---

## License

[BSD-3-Clause](./LICENSE) — chosen to align with `tlparse` and the broader PyTorch ecosystem.
