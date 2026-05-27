"""Command-line entry point for compile-lens (the ``cl`` console script).

This is a thin dispatcher (design.md P7): the real work lives in the library
(``compile_lens`` collectors) and the Rust analyzers. For the v0.5.0 skeleton the
subcommands are placeholders that report "not implemented yet"; each lands in a later phase.
"""

from __future__ import annotations

import argparse
import sys

from compile_lens import __version__

#: Subcommand name -> one-line help. Each is wired up in a later phase.
_SUBCOMMANDS: dict[str, str] = {
    "session": "Run all collectors and emit a unified report (hero entry)",
    "collect": "Collect a .cls.json session artifact from a run / logs / tlparse output",
    "recompile-summary": "Summarize recompilation causes (Tool 1)",
    "diff": "Diff two compile artifacts (Tool 2, WL-signature)",
    "divergence": "Locate eager-vs-compile numerical divergence (Tool 3)",
    "compile-lint": "Lint for torch.compile anti-patterns (Tool 4)",
    "kernel-roofline": "Kernel-level roofline triage (Tool 5)",
    "scrub": "Sanitize an artifact / report before sharing",
    "migrate": "Migrate an older .cls.json to the current schema",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``cl`` argument parser with placeholder subcommands."""
    parser = argparse.ArgumentParser(
        prog="cl",
        description="compile-lens — diagnostics for torch.compile production observability.",
    )
    parser.add_argument("--version", action="version", version=f"compile-lens {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, help_text in _SUBCOMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(
        f"cl {args.command}: not implemented yet (v{__version__} skeleton).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
