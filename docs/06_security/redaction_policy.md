# compile-lens Redaction Policy

**Related**: `threat_model.md` (same directory), `../../SECURITY.md`

---

## 1. Policy levels

The `session.redaction_policy` field in the `.cls.json` schema enumerates 4 levels:

| Level | Default for | Paths | Command | Source excerpt | Kernel source | Host | Shareable? |
|---|---|---|---|---|---|---|---|
| `public-safe` | user `cl scrub` for Show HN / public README | fully anonymized | nulled | omitted | omitted | nulled | ✅ |
| `default-strict` | **DEFAULT (no flag)** | normalized to relative | token-scrubbed | omitted | omitted | FQDN-hashed | ✅ |
| `internal` | user passes `--redaction internal` (team-internal sharing) | relative | token-scrubbed | opt-in field | opt-in field | retained | ✅ within a trusted team |
| `confidential` | user passes `--redaction confidential` (no intent to share) | raw | raw | raw | raw | raw | ❌ HTML watermark |

**Rule**: in a chained policy, **the strictest level always wins** (`cl scrub --to
public-safe` always works, regardless of the source level).

---

## 2. Default-strict redaction rules

### 2.1 Path normalization

Applies to: `session.command` (argv[0]), `lint_findings[].source_location.file`,
`kernels[].source_path`, `kernels[].ptx_path`, and all path-typed fields in the artifact.

**Rules** (applied in order, first match wins):

| Pattern (regex) | Replacement | Note |
|---|---|---|
| `^/home/[^/]+/(.+/)?<repo>/(.*)$` | `{repo}/$2` | `$repo` = detected git repo basename |
| `^/Users/[^/]+/(.+/)?<repo>/(.*)$` | `{repo}/$2` | macOS variant |
| `^/opt/conda/envs/[^/]+/lib/python[\d.]+/site-packages/torch/(.*)$` | `{torch_install}/$1` | PyTorch install path |
| `^/opt/conda/envs/[^/]+/lib/python[\d.]+/site-packages/triton/(.*)$` | `{triton_install}/$1` | Triton install path |
| `^/usr/local/lib/python[\d.]+/site-packages/torch/(.*)$` | `{torch_install}/$1` | system Python variant |
| `^.*\.cache/torch_extensions/(.*)$` | `{torch_cache}/$1` | torch extension cache dir |
| `^.*\.cache/torch_compile/(.*)$` | `{torch_compile_cache}/$1` | torch.compile cache |
| `^/tmp/(.*)$` | `/tmp/$1` (unchanged) | tmp dir leaks little |
| (any other absolute path) | unchanged + WARN | `cl scrub --dry-run` flags these |

**Repo detection**: walk up from the current directory to find a `.git` dir. Its basename is
the `{repo}` token.

**Edge case**: a path under `~/.ssh/`, `~/.aws/`, `~/.gnupg/` → **refuse to record** (the
collector raises `ClsError::SensitivePathDetected` with `CLS-E0008`).

### 2.2 Argv token scrubbing

Applies to: `session.command`.

**Regex patterns** (scrub the value, keep the key):

```text
--api-key=\S+                  → --api-key=<scrubbed>
--api[_-]?key=\S+              → --api-key=<scrubbed>
--token=\S+                    → --token=<scrubbed>
--hf-token=\S+                 → --hf-token=<scrubbed>
--wandb-key=\S+                → --wandb-key=<scrubbed>
--password=\S+                 → --password=<scrubbed>
--secret=\S+                   → --secret=<scrubbed>
--auth=\S+                     → --auth=<scrubbed>
\bBearer\s+[A-Za-z0-9._-]+     → Bearer <scrubbed>
\bhf_[A-Za-z0-9]{20,}          → hf_<scrubbed>     # HuggingFace token format
\bsk-[A-Za-z0-9]{20,}          → sk-<scrubbed>     # OpenAI key format
\bAKIA[A-Z0-9]{16}             → AKIA<scrubbed>    # AWS access key
\bxox[abopr]-[A-Za-z0-9-]+     → xox<scrubbed>     # Slack token
```

**Strict mode** (`cl scrub --strict` or `--redaction public-safe`): additionally scrub any
argv value matching `[A-Za-z0-9+/]{32,}` (long base64-looking) or `[A-Fa-f0-9]{32,}` (long
hex) as a **possible secret**.

### 2.3 Source excerpt — omitted by default

```json
{
  "source_location": {
    "file": "{repo}/model.py",
    "line_start": 47,
    "line_end": 49,
    "code_excerpt": null         // ← omitted by default
  }
}
```

**Opt-in**: a user passes `--include-source` to include the excerpt. Even then, the excerpt
is limited to the lines the finding explicitly references (no surrounding context).

### 2.4 Kernel source — omitted by default

```json
{
  "kernels": [{
    "name": "fused_attention_qkv",
    "source_path": "{torch_cache}/...",      // path-normalized
    "ptx_path": "<scrubbed>",                 // PTX always omitted under default-strict
    "kernel_source_excerpt": null,           // ← omitted by default
    "launch_config": {...},                  // ← retained (no IP)
    "features": {                            // ← retained (aggregate stats only)
      "flops": 1.2e10,
      "bytes_loaded": 8.4e8,
      "num_regs": 168,
      "n_spills": 0
    },
    "measurements": {...}                    // ← retained
  }]
}
```

**Rationale**: kernel source / PTX **is** the IP of a fine-tuned production model (numerical
hyperparameters, fusion-strategy choices). Launch config and aggregate features are less
sensitive — they describe *shape*, not *content*.

**Opt-in**: `--include-kernel-source` for the `internal` policy and above.

### 2.5 Host

| Level | `session.host` |
|---|---|
| `default-strict` | `<fqdn-hash>` (e.g. `dh-7f2a91b3`) — a stable per-machine identifier that does not leak the FQDN |
| `internal` | full FQDN |
| `confidential` | full FQDN |
| `public-safe` | `null` |

FQDN hash = `sha256(fqdn + per-install-salt)[:8]`. The per-install salt lives in
`~/.compile-lens/install-id` and is not shared.

### 2.6 Environment variables

`session.env_snapshot.relevant_env_vars` is allowlisted at collection time (only known
torch.compile-relevant vars). It **never** captures `*_KEY`, `*_TOKEN`, `*_SECRET`,
`*_PASSWORD`, even if a user explicitly sets one as torch.compile-relevant.

Allowlist (case-insensitive prefix match against the env var name):
```text
TORCH_*
TORCHINDUCTOR_*
TORCHDYNAMO_*
TRITON_*
CUDA_*
CUDNN_*
NCCL_DEBUG
HSA_*       (AMD)
```

**Explicit denylist** (even if matched above):
```text
*_KEY, *_TOKEN, *_SECRET, *_PASSWORD, *_AUTH, *_CREDENTIAL
HF_TOKEN, HUGGINGFACE_TOKEN, WANDB_API_KEY, OPENAI_API_KEY, etc.
```

---

## 3. `cl scrub` CLI semantics

### 3.1 Modes

```bash
# Default: in-place sanitize the artifact (overwrite)
$ cl scrub session.cls.json
✅ Wrote sanitized session.cls.json (12 paths normalized, 2 argv tokens scrubbed, 47 kernel sources removed)

# To public-safe (more aggressive)
$ cl scrub session.cls.json --to public-safe --output session-public.cls.json
✅ Wrote session-public.cls.json (host nulled, full strict redaction applied)

# Dry-run (no write; show what would happen)
$ cl scrub session.cls.json --dry-run
Would redact:
  - session.command: "python /home/jdoe/proj/megacorp-llm/train.py --hf-token=hf_abc123"
    → "python {repo}/train.py --hf-token=<scrubbed>"
  - session.host: "ml-prod-07.megacorp.internal" → "<fqdn-hash:dh-7f2a91b3>"
  - 12 kernel.source_path entries (PyTorch cache paths normalized)
  - 47 kernel.ptx_path entries (scrubbed)
  - 8 kernel.kernel_source_excerpt entries (omitted)
  - 23 lint_finding.source_location.file entries (normalized to {repo}/)

# Sanitize an HTML report
$ cl scrub report.html --output report-public.html
✅ Wrote report-public.html (CSP applied, 4 XSS vectors escaped)

# Strict mode (catch novel secrets)
$ cl scrub session.cls.json --strict
✅ Wrote session.cls.json (default-strict + 3 long-string argv values scrubbed as potential secrets)
```

### 3.2 Verification mode

```bash
# Audit whether an artifact is share-safe
$ cl scrub --verify session-public.cls.json
✅ Artifact policy: default-strict
✅ No raw paths detected
✅ No raw tokens detected in command
✅ No source excerpts present
✅ No kernel sources present
✅ Host is FQDN-hashed
→ SHARE-SAFE for public/team distribution

# Or:
$ cl scrub --verify session.cls.json --target public-safe
⚠️ Artifact policy: default-strict (target: public-safe)
⚠️ Host is FQDN-hashed but not nulled (public-safe requires null)
⚠️ 2 paths under /tmp/ retained (public-safe should normalize to {tmp}/)
→ NOT public-safe; run: cl scrub --to public-safe
```

### 3.3 Batch operations

```bash
$ cl scrub ./session-archive/*.cls.json --to public-safe --output ./session-archive-public/
✅ Processed 47 files, all share-safe.
```

---

## 4. Policy upgrade / downgrade rules

**Upgrade (stricter)**: always allowed.
- `confidential` → `internal` → `default-strict` → `public-safe`
- a lossy transform of raw data; redacted data cannot be restored

**Downgrade (less strict)**: **refused**.
- a `default-strict` artifact cannot become `internal` (that would require re-collecting raw data)
- `cl scrub --to internal session-default-strict.cls.json` returns
  `CLS-E0009: cannot demote redaction policy; re-collect with --redaction=internal`

**Cross-level reads**: an analyzer reading a `default-strict` artifact behaves as if all
opt-in fields are absent (because they genuinely are). There is no "secretly preserve raw" path.

---

## 5. UX: where redaction is surfaced

### 5.1 At collection time

```bash
$ cl session  # default-strict by default, prints once:
ℹ️  Collecting with redaction_policy=default-strict. Use --redaction internal for full data within a trusted team.
ℹ️  cl scrub --dry-run session.cls.json to preview share-time sanitization.
```

### 5.2 Report generation

`cl session report` emits HTML with a header banner:

```html
<div class="cls-policy-banner cls-policy-default-strict">
  📋 Redaction: default-strict (paths normalized, tokens scrubbed, source/kernel sources omitted)
  <a href="https://compile-lens.dev/docs/06_security/redaction_policy">What does this mean?</a>
</div>
```

For `confidential`:

```html
<div class="cls-policy-banner cls-policy-confidential">
  ⛔ CONFIDENTIAL — DO NOT SHARE. Raw paths / source / kernel sources retained.
  Run: cl scrub <file> to sanitize before distribution.
</div>
```

### 5.3 MCP exposure

Phase 2: the agent receives `redaction_policy` as metadata on each finding. The agent SHOULD
NOT include omitted fields in a user-facing answer ("I cannot retrieve the source excerpt
because the artifact is `default-strict`; ask the user to re-collect with `--redaction
internal` if needed").

---

## 6. Test discipline (CI)

The `cls-scrub` crate has a full test corpus:

```
crates/cls-scrub/test_corpus/
├── before/
│   ├── session-leaks-paths.cls.json       # has /home/jdoe/...
│   ├── session-leaks-tokens.cls.json      # has --api-key=...
│   ├── session-leaks-source.cls.json      # has a source excerpt
│   └── report-xss-vectors.html            # has <script> injection attempts
└── after/
    └── (expected scrubbed outputs, byte-identical compare)
```

**CI assertions**:
1. Each `before/` file, after `cl scrub`, is byte-identical to its `after/` counterpart.
2. Each `after/` file passes `cl scrub --verify`.
3. Fuzz test: random ASCII strings injected into argv / paths / op-name → the output never
   contains the original raw string.
4. Property test: scrubbing is idempotent (`scrub(scrub(x)) == scrub(x)`).

**Release blocker**: if any `cls-scrub` test fails, or the test corpus is incomplete (each
new pattern needs a `before/after/` pair), the release is blocked.

---

## 7. Open questions

| # | Question | Note |
|---|---|---|
| 1 | Should `default-strict` scrub the stack trace in `lint_findings[].diagnostic_chain`? | Likely yes — stack traces contain absolute paths. Defer to post-MVP if not in initial scope. |
| 2 | How to handle non-UTF-8 paths (Windows)? | Defer — Windows is not an MVP priority; assume Linux/macOS. |
| 3 | Should `cl scrub` rewrite in place or always require `--output`? | Currently defaults to in-place; consider a safer default (always `--output`) post-user-feedback. |
| 4 | Provide a pre-commit hook example? | Yes, add to `docs/06_security/` post-MVP: a `.pre-commit-config.yaml` that auto-scrubs on commit. |
| 5 | Support a diff format for sanitized vs raw artifacts? | `cl scrub --diff` could show before/after side-by-side — defer post-MVP. |

---

## 8. References

- PyTorch `tlparse` warning about trace contents: <https://docs.pytorch.org/...>
- MCP spec security section: <https://modelcontextprotocol.io/specification/security>
- OWASP Top 10 (2023) — A03 Injection, A06 Vulnerable Components
- the `ammonia` Rust crate for HTML sanitization
