"""compile-lens — diagnostics for torch.compile production observability."""

from compile_lens.tools.divergence import DivergenceSession, divergence_session

__version__ = "0.5.0"

__all__ = ["DivergenceSession", "divergence_session", "__version__"]
