"""Default-strict redaction applied **at collection time** (discipline D11).

A ``default-strict`` ``.cls.json`` must never contain raw secrets to begin with — redaction
is a property of *how the collector writes*, not a post-hoc scrub. This module is the set of
pure redaction primitives (spec: ``docs/06_security/redaction_policy.md`` §2); the collector
calls them in ``finalize`` when the session's policy is strict.

- :func:`scrub_command` — token-scrub an argv string (keep the flag, redact the value).
- :func:`normalize_path` — collapse user/install paths to ``{repo}`` / ``{torch_install}``
  placeholders; **refuse** a path under ``~/.ssh`` / ``~/.aws`` / ``~/.gnupg`` by raising
  :class:`SensitivePathError` (``CLS-E0008``).
- :func:`hash_host` — replace an FQDN with a stable ``dh-<8hex>`` per-machine identifier.
- :func:`filter_env_vars` — whitelist torch.compile-relevant env vars, denylist secrets.
- :func:`is_strict` — whether a policy redacts by default.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path

from compile_lens._schema import RedactionPolicy

#: A path under any of these (after ``~`` expansion) is refused, never recorded.
_SENSITIVE_DIRS = ("/.ssh/", "/.aws/", "/.gnupg/")


class SensitivePathError(Exception):
    """A collected path points inside a credential directory; refuse to record it.

    Mirrors the Rust ``CLS-E0008`` variant so both halves of the toolkit speak one error
    vocabulary (D8). Carries the offending path for the diagnostic.
    """

    code = "CLS-E0008"

    def __init__(self, path: str) -> None:
        super().__init__(
            f"{self.code}: refusing to record a path inside a credential directory: {path!r} "
            f"(re-run from outside ~/.ssh, ~/.aws, ~/.gnupg)"
        )
        self.path = path


# ── command (argv) token scrubbing — §2.2 ────────────────────────────────────────────────
# (pattern, replacement) applied in order; keep the key, redact the value.
_COMMAND_SCRUBS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), r)
    for p, r in (
        (r"(--api[_-]?key=)\S+", r"\1<scrubbed>"),
        (r"(--hf-token=)\S+", r"\1<scrubbed>"),
        (r"(--wandb-key=)\S+", r"\1<scrubbed>"),
        (r"(--token=)\S+", r"\1<scrubbed>"),
        (r"(--password=)\S+", r"\1<scrubbed>"),
        (r"(--secret=)\S+", r"\1<scrubbed>"),
        (r"(--auth=)\S+", r"\1<scrubbed>"),
        (r"\bBearer\s+[A-Za-z0-9._-]+", "Bearer <scrubbed>"),
        (r"\bhf_[A-Za-z0-9]{20,}", "hf_<scrubbed>"),
        (r"\bsk-[A-Za-z0-9]{20,}", "sk-<scrubbed>"),
        (r"\bAKIA[A-Z0-9]{16}", "AKIA<scrubbed>"),
        (r"\bxox[abopr]-[A-Za-z0-9-]+", "xox<scrubbed>"),
    )
)


def scrub_command(command: str) -> str:
    """Redact secret-looking tokens from an argv string, preserving the flag/key."""
    for pattern, replacement in _COMMAND_SCRUBS:
        command = pattern.sub(replacement, command)
    return command


# ── path normalization — §2.1 ─────────────────────────────────────────────────────────────
_SITE_PACKAGES = r"/lib/python[\d.]+/site-packages"
_PATH_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), r)
    for p, r in (
        (rf"^/opt/conda/envs/[^/]+{_SITE_PACKAGES}/torch/(.*)$", r"{torch_install}/\1"),
        (rf"^/opt/conda/envs/[^/]+{_SITE_PACKAGES}/triton/(.*)$", r"{triton_install}/\1"),
        (rf"^/usr/local{_SITE_PACKAGES}/torch/(.*)$", r"{torch_install}/\1"),
        (r"^.*\.cache/torch_extensions/(.*)$", r"{torch_cache}/\1"),
        (r"^.*\.cache/torch_compile/(.*)$", r"{torch_compile_cache}/\1"),
    )
)


def normalize_path(path: str, *, repo: str | None = None) -> str:
    """Collapse a path to a non-identifying placeholder form.

    Raises :class:`SensitivePathError` if the path is inside a credential directory. A
    ``{repo}`` basename, when provided, collapses ``/home/<user>/…/<repo>/<rest>`` (and the
    macOS ``/Users`` variant) to ``{repo}/<rest>``. Unrecognized absolute paths are returned
    unchanged (``cl scrub --dry-run`` flags those).
    """
    if any(d in path for d in _SENSITIVE_DIRS):
        raise SensitivePathError(path)

    if repo:
        repo_rule = re.compile(rf"^/(?:home|Users)/[^/]+/(?:.+/)?{re.escape(repo)}/(.*)$")
        m = repo_rule.match(path)
        if m:
            return f"{{repo}}/{m.group(1)}"

    for pattern, replacement in _PATH_RULES:
        new = pattern.sub(replacement, path)
        if new != path:
            return new
    return path


# ── host FQDN hashing — §2.5 ──────────────────────────────────────────────────────────────
def hash_host(fqdn: str, *, salt: str) -> str:
    """Stable per-machine identifier ``dh-<8hex>`` = ``sha256(fqdn + salt)[:8]``.

    The salt is a per-install secret (``~/.compile-lens/install-id``), so the hash is stable
    on one machine but does not leak the FQDN and is not cross-machine correlatable.
    """
    digest = hashlib.sha256((fqdn + salt).encode("utf-8")).hexdigest()
    return f"dh-{digest[:8]}"


def install_salt() -> str:
    """Per-install salt for :func:`hash_host`.

    ``CLS_INSTALL_ID`` env var wins (deterministic for tests / reproducible runs); otherwise
    read-or-create ``~/.compile-lens/install-id`` (a random hex secret that is never shared).
    """
    override = os.environ.get("CLS_INSTALL_ID")
    if override:
        return override
    path = Path.home() / ".compile-lens" / "install-id"
    if path.exists():
        return path.read_text().strip()
    salt = secrets.token_hex(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(salt)
    return salt


# ── env var filtering — §2.6 ──────────────────────────────────────────────────────────────
_ENV_WHITELIST_PREFIXES = (
    "TORCH_",
    "TORCHINDUCTOR_",
    "TORCHDYNAMO_",
    "TRITON_",
    "CUDA_",
    "CUDNN_",
    "HSA_",
)
_ENV_WHITELIST_EXACT = ("NCCL_DEBUG",)
_ENV_DENY_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_AUTH", "_CREDENTIAL")
_ENV_DENY_EXACT = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY")


def filter_env_vars(env: Mapping[str, str]) -> dict[str, str]:
    """Keep only torch.compile-relevant env vars; drop secrets even if they match a prefix.

    The denylist wins over the whitelist (``TORCH_API_KEY`` is dropped), so a secret can
    never be captured by widening the whitelist.
    """
    out: dict[str, str] = {}
    for key, value in env.items():
        upper = key.upper()
        if upper in _ENV_DENY_EXACT or any(upper.endswith(s) for s in _ENV_DENY_SUFFIXES):
            continue
        whitelisted = upper in _ENV_WHITELIST_EXACT or any(
            upper.startswith(p) for p in _ENV_WHITELIST_PREFIXES
        )
        if whitelisted:
            out[key] = value
    return out


# ── policy ────────────────────────────────────────────────────────────────────────────────
def is_strict(policy: RedactionPolicy) -> bool:
    """Whether the policy redacts by default (``default-strict`` / ``public-safe``).

    ``internal`` / ``confidential`` keep raw fields (opt-in / trusted), so the collector does
    not auto-scrub under them.
    """
    return policy in (RedactionPolicy.DEFAULT_STRICT, RedactionPolicy.PUBLIC_SAFE)
