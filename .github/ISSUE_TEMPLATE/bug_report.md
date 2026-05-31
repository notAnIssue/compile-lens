---
name: Bug report
about: A defect in compile-lens — something that does not work as documented
title: "[bug] "
labels: ["bug", "triage"]
---

<!--
Before opening:
  - Search closed issues / discussions — bugs are often known and worked-around.
  - If the bug exposes a SECURITY issue (e.g. unscrubbed sensitive data in a generated artifact),
    do NOT file a public issue. Follow SECURITY.md instead.
-->

## What happened

<!-- One paragraph. Include the exact command or Python snippet you ran. -->

```text
$ cl <subcommand> ...
<output / stack trace>
```

## What you expected

<!-- A line or two. -->

## Environment

- `cl --version`:
- Python version (`python --version`):
- PyTorch version (`python -c "import torch; print(torch.__version__)"`):
- OS / distro:
- CUDA / GPU (if applicable):

## Minimal reproducer

<!-- A self-contained script ideally; otherwise the smallest path that triggers it. -->

```python
import torch
import compile_lens as cl
# ...
```

## Anything else?

<!-- Logs, screenshots, related ADRs / issues. -->
