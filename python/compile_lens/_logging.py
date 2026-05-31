"""Structured logging setup (structlog) for the Python half of compile-lens.

The Rust binary configures `tracing` from `CLS_LOG` / `CLS_LOG_FORMAT`; the Python
front-end mirrors the same surface here so a single set of environment knobs controls
verbosity on both sides of the subprocess boundary (the design doc / discipline D9 —
tracing + debug only; no metric export).

Environment variables read by :func:`configure_logging`:

- ``CLS_LOG`` — log level, case-insensitive (``DEBUG`` / ``INFO`` / ``WARNING`` /
  ``ERROR``). Default: ``WARNING``.
- ``CLS_DEBUG=1`` — force ``DEBUG`` verbosity (overrides ``CLS_LOG``). The Python-side
  shortcut for ad-hoc deep dives without remembering the level name.
- ``CLS_LOG_FORMAT=json`` — emit one JSON object per event (opt-in). Anything else
  (incl. unset) gives the human-readable :class:`structlog.dev.ConsoleRenderer`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import structlog


def _resolve_level() -> int:
    """Translate ``CLS_DEBUG`` / ``CLS_LOG`` into a :mod:`logging` level int.

    ``CLS_DEBUG=1`` wins outright (the Python-side "give me everything" idiom); otherwise
    ``CLS_LOG`` is parsed case-insensitively. An unknown level name falls back to
    ``WARNING`` rather than raising — a logging misconfiguration must not crash a
    diagnostics tool.
    """
    if os.environ.get("CLS_DEBUG") == "1":
        return logging.DEBUG
    name = os.environ.get("CLS_LOG", "WARNING").upper()
    return getattr(logging, name, logging.WARNING)


def configure_logging() -> None:
    """Install a process-wide structlog configuration from the ``CLS_*`` env vars.

    Idempotent in practice (calling twice replaces the configuration), so safe to invoke
    near the CLI entry point.
    """
    level = _resolve_level()
    use_json = os.environ.get("CLS_LOG_FORMAT", "").lower() == "json"

    final_renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            final_renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
