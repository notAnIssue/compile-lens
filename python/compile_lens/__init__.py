"""compile-lens — diagnostics for torch.compile production observability."""

from compile_lens.capture import CaptureResult, capture
from compile_lens.session import Session, session
from compile_lens.tools.divergence import (
    CausalAttribution,
    DivergenceFindings,
    DivergenceSession,
    MinifierStatus,
    accuracy_minifier,
    attribute_divergence,
    divergence_session,
)

__version__ = "0.5.0"

__all__ = [
    "CaptureResult",
    "CausalAttribution",
    "DivergenceFindings",
    "DivergenceSession",
    "MinifierStatus",
    "Session",
    "accuracy_minifier",
    "attribute_divergence",
    "capture",
    "divergence_session",
    "session",
    "__version__",
]
