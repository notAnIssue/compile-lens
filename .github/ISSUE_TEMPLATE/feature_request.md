---
name: Feature request
about: A new tool, capability, or workflow inside compile-lens' scope
title: "[feat] "
labels: ["enhancement", "triage"]
---

<!--
compile-lens has a sharply-defined scope (see the design doc "Not this"). It is *not*
a replacement compiler / fuzzer / correctness oracle, and it does not rewrite the
official PyTorch tracing tools — it interprets their output. If the idea is outside
that scope it will (politely) be closed; that is a feature of the project, not a slight.
-->

## What is the problem?

<!-- What user goal is unmet today? Skip implementation ideas for now. -->

## Where in the toolkit does it live?

<!-- Pick one or sketch a new tool. the design doc lists the existing tools. -->

- [ ] Tool 1 (recompile aggregator)
- [ ] Tool 2 (compile diff)
- [ ] Tool 2b (cache stability)
- [ ] Tool 3 (divergence detector)
- [ ] Tool 4 (compile-lint)
- [ ] Tool 5 (kernel roofline triage)
- [ ] Tool 6 (CODA fusion detector)
- [ ] Hero (`cl.session()` / HTML report)
- [ ] New tool — describe below

## Proposed shape (optional)

<!-- A few bullet points on what the UX could look like. Pseudocode is fine. -->

## Out of scope?

<!-- Briefly check it against the design doc -->
