"""compile-lens — diagnostics for torch.compile production observability."""

from compile_lens.tools.divergence import (
    DivergenceFindings,
    DivergenceSession,
    MinifierStatus,
    accuracy_minifier,
    divergence_session,
)

__version__ = "0.5.0"

__all__ = [
    "DivergenceFindings",
    "DivergenceSession",
    "MinifierStatus",
    "accuracy_minifier",
    "divergence_session",
    "__version__",
]
