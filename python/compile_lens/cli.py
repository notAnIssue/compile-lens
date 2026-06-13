"""Command-line entry point for compile-lens (the ``cl`` console script).

This is a thin dispatcher (design.md P7): the real work lives in the library
(``compile_lens`` collectors) and the Rust analyzers. Most subcommands are still
placeholders that report "not implemented yet"; each lands in a later phase. The wired
one is ``collect`` (Tool 1), which drives :class:`RecompileCollector` to write a
``.cls.json`` the Rust ``cl recompile-summary`` then analyzes.
"""

from __future__ import annotations

import argparse
import sys
import tomllib

from compile_lens import __version__
from compile_lens._schema import RedactionPolicy

#: Subcommand name -> one-line help. The ones not yet wired print "not implemented yet".
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


def _add_collect_args(sp: argparse.ArgumentParser) -> None:
    """Wire ``cl collect``'s flags — the mutually-exclusive mode group + output + redaction,
    mirroring the Rust ``cl collect`` surface (design.md ADR-006: the Python ``cl`` is the
    user-facing front-end; the Rust binary is invoked for analysis)."""
    mode = sp.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-logs", metavar="PATH", help="Mode A — a TORCH_LOGS=recompiles capture"
    )
    mode.add_argument("--from-tlparse", metavar="DIR", help="Mode B — a tlparse output directory")
    mode.add_argument(
        "--from-dynamo-explain",
        action="store_true",
        help="Mode C — programmatic torch._dynamo.explain (Python API only; see error text)",
    )
    sp.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        required=True,
        help="where to write the .cls.json artifact",
    )
    sp.add_argument(
        "--redaction",
        choices=[p.value for p in RedactionPolicy],
        default=RedactionPolicy.DEFAULT_STRICT.value,
        help="redaction policy applied at capture time (default: default-strict)",
    )


def _add_compile_lint_args(sp: argparse.ArgumentParser) -> None:
    """Wire ``cl compile-lint``'s flags. The Python front-end *scans* source into a ``.cls.json``;
    the Rust ``cl compile-lint <artifact>`` then analyzes it against the correctness database
    (design.md ADR-006; the unified single-command front-end is the Phase 7 hero)."""
    sp.add_argument("path", metavar="PATH", help="a .py file, or a directory of .py files, to scan")
    sp.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        required=True,
        help="where to write the .cls.json artifact",
    )
    sp.add_argument(
        "--db",
        metavar="PATH",
        help="correctness-database TOML; its operator-family rules drive detection. Without it, "
        "only the structural in_place_op_on_alias pattern fires.",
    )
    sp.add_argument(
        "--redaction",
        choices=[p.value for p in RedactionPolicy],
        default=RedactionPolicy.DEFAULT_STRICT.value,
        help="redaction policy applied at capture time (default: default-strict)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``cl`` argument parser. ``collect`` and ``compile-lint`` carry real
    flags; the other subcommands are bare placeholders until their phase."""
    parser = argparse.ArgumentParser(
        prog="cl",
        description="compile-lens — diagnostics for torch.compile production observability.",
    )
    parser.add_argument("--version", action="version", version=f"compile-lens {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, help_text in _SUBCOMMANDS.items():
        sp = subparsers.add_parser(name, help=help_text)
        if name == "collect":
            _add_collect_args(sp)
        elif name == "compile-lint":
            _add_compile_lint_args(sp)
    return parser


def _run_collect(args: argparse.Namespace) -> int:
    """Drive :class:`RecompileCollector` for ``cl collect`` and write the artifact.

    Mode A (``--from-logs``) and Mode B (``--from-tlparse``) take a filesystem path and run
    here. Mode C (``--from-dynamo-explain``) needs a live ``torch._dynamo.explain`` *result*
    object, which the CLI cannot produce — it is programmatic-only.
    """
    from compile_lens.collectors.recompile import Mode, RecompileCollector

    if args.from_dynamo_explain:
        print(
            "cl collect --from-dynamo-explain is programmatic-only: build a "
            "torch._dynamo.explain(...) result and pass it to "
            "RecompileCollector(...).from_dynamo_explain(result). "
            "The CLI supports --from-logs and --from-tlparse.",
            file=sys.stderr,
        )
        return 2

    if args.from_logs is not None:
        mode, source = Mode.LOGS, args.from_logs
    else:
        mode, source = Mode.TLPARSE, args.from_tlparse

    # Record the invocation for provenance; under a strict policy finalize() scrubs it (D11).
    collector = RecompileCollector(
        args.output, redaction_policy=args.redaction, command=" ".join(sys.argv)
    )
    try:
        collector.collect(mode, source)
        written = collector.finalize()
    except FileNotFoundError as exc:
        print(f"cl collect: input not found — {exc}", file=sys.stderr)
        return 3

    print(f"wrote {written}")
    return 0


def _run_compile_lint(args: argparse.Namespace) -> int:
    """Scan ``args.path`` for Tool 4 anti-patterns and write the candidate ``.cls.json``.

    This is the scan half (Layer-1, static AST); run the Rust ``cl compile-lint <output>`` to
    analyze it against the correctness database and gate CI on its exit code (design.md ADR-006).
    """
    from compile_lens.collectors.lint_collect import LintCollector
    from compile_lens.correctness_db import load_operator_rules

    # The database's [detector] rules drive operator-family detection (ADR-036); without --db only
    # the structural pattern fires. The same database is passed to the Rust analyzer for evidence.
    try:
        rules = load_operator_rules(args.db) if args.db else None
    except (FileNotFoundError, tomllib.TOMLDecodeError, KeyError) as exc:
        print(f"cl compile-lint: could not read --db — {exc}", file=sys.stderr)
        return 3

    collector = LintCollector(args.output, operator_rules=rules, redaction_policy=args.redaction)
    try:
        count = collector.scan_path(args.path)
    except FileNotFoundError as exc:
        print(f"cl compile-lint: source not found — {exc}", file=sys.stderr)
        return 3
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"cl compile-lint: could not parse source — {exc}", file=sys.stderr)
        return 3

    written = collector.finalize()
    print(f"wrote {written} ({count} candidate finding(s))")
    print(f"  analyze it with: cl compile-lint {written}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "collect":
        return _run_collect(args)
    if args.command == "compile-lint":
        return _run_compile_lint(args)
    print(
        f"cl {args.command}: not implemented yet (v{__version__} skeleton).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
