"""Structured logging setup (structlog).

Skeleton only: a later phase (S0.16) wires ``CLS_LOG`` verbosity, ``CLS_DEBUG`` verbose
mode, and JSON output (the design doc / discipline D9 — tracing + debug only, no metric export).
"""

from __future__ import annotations

import structlog


def configure_logging() -> None:
    """Configure structlog with plain-text, human-readable defaults."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )
