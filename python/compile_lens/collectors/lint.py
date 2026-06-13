"""Layer-1 AST lint scanner for Tool 4 (`compile-lint`).

This is the cheap, trace-free pre-filter (ADR-032's two-layer design): it reads the user's source
with the standard :mod:`ast` module and flags structural anti-patterns known to correlate with
`torch.compile` correctness divergence. It **never executes the user's code** — it only parses it,
so it is safe to run in CI on untrusted source. The high-precision Layer 2 (functionalized FX graph
confirmation) lands separately; this layer produces candidates.

v0 covers the two highest-frequency patterns from Li et al. 2026:

- ``in_place_op_on_alias`` (§3.2.3) — an in-place write through a tensor that aliases another
  (e.g. ``y = x.expand(...); y[0] = 1``). Purely structural; no configuration.
- ``operator_non_default_param`` (§3.2.2) — a call to a *watched* operator with a non-default value
  for a *watched* parameter. The detector logic ships here; the specific operator/parameter pairs
  are data, supplied by the correctness database (so this layer carries no hard-coded bug claims).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

#: Tensor methods that return a **view/alias** sharing storage with the source (not a copy). A copy
#: maker like ``clone`` is deliberately absent — writing through a clone is safe.
_ALIASING_METHODS = frozenset(
    {
        "view",
        "expand",
        "expand_as",
        "reshape",
        "transpose",
        "permute",
        "t",
        "narrow",
        "select",
        "squeeze",
        "unsqueeze",
        "broadcast_to",
        "swapaxes",
        "movedim",
    }
)


@dataclass(frozen=True)
class LintHit:
    """One pattern match, with its source location."""

    pattern_name: str
    line: int
    col: int
    message: str


class LintPatternScanner:
    """Scan Python source for Tool 4's anti-patterns, purely via the AST.

    ``operator_rules`` drives the operator-family patterns (ADR-036): it maps an operator name to a
    ``{parameter_name: pattern_category}`` table. When the scanner sees that operator called with a
    watched parameter, it emits a candidate whose ``pattern_category`` is **that pattern's name** —
    so the analyzer can look the exact pattern up in the correctness database. The real rules are
    supplied by the database (via the front-end); it defaults to empty, so with no rules the
    operator family matches nothing while the structural ``in_place_op_on_alias`` is always active.
    """

    def __init__(self, operator_rules: dict[str, dict[str, str]] | None = None) -> None:
        # operator -> {param_name -> pattern_category}; copied so the caller's dicts aren't aliased.
        self.operator_rules: dict[str, dict[str, str]] = (
            {op: dict(params) for op, params in operator_rules.items()}
            if operator_rules is not None
            else {}
        )

    def scan(self, source: str, filename: str = "<unknown>") -> list[LintHit]:
        # Parse only — the user's code is never compiled-and-run, so scanning is side-effect free.
        tree = ast.parse(source, filename=filename)
        aliases = self._collect_aliases(tree)
        hits: list[LintHit] = []
        hits.extend(self._find_in_place_on_alias(tree, aliases))
        hits.extend(self._find_operator_non_default_param(tree))
        return _apply_suppressions(hits, source, tree)

    # ── alias tracking ──────────────────────────────────────────────────────────────────

    def _collect_aliases(self, tree: ast.AST) -> set[str]:
        """Names bound to an aliasing expression. A two-pass simplification (no scope/reassign
        modelling) — enough for the common straight-line case; richer alias sources are post-v0."""
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and self._is_aliasing_expr(node.value)
            ):
                aliases.add(node.targets[0].id)
        return aliases

    @staticmethod
    def _is_aliasing_expr(value: ast.expr) -> bool:
        # `base.view(...)` / `.expand(...)` etc.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            return value.func.attr in _ALIASING_METHODS
        # `base.T` — a transpose view.
        if isinstance(value, ast.Attribute) and value.attr == "T":
            return True
        return False

    # ── in_place_op_on_alias ────────────────────────────────────────────────────────────

    def _find_in_place_on_alias(self, tree: ast.AST, aliases: set[str]) -> list[LintHit]:
        hits: list[LintHit] = []
        for node in ast.walk(tree):
            found = self._in_place_target(node)
            if found is not None and found[0] in aliases:
                base, line, col = found
                hits.append(
                    LintHit(
                        "in_place_op_on_alias",
                        line,
                        col,
                        f"in-place op on `{base}`, which aliases another tensor (Li et al. §3.2.3)",
                    )
                )
        return hits

    @staticmethod
    def _in_place_target(node: ast.AST) -> tuple[str, int, int] | None:
        """``(base_var, line, col)`` of an in-place op, or ``None`` if this node is not one. The
        position is read here, where ``node`` is narrowed to a type that carries one."""
        # `name[...] = ...`
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    return target.value.id, node.lineno, node.col_offset
        # `name[...] += ...`
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Subscript)
            and isinstance(node.target.value, ast.Name)
        ):
            return node.target.value.id, node.lineno, node.col_offset
        # `name.add_(...)` — a trailing-underscore method that is not a dunder.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if (
                attr.endswith("_")
                and not attr.startswith("__")
                and isinstance(node.func.value, ast.Name)
            ):
                return node.func.value.id, node.lineno, node.col_offset
        return None

    # ── operator_non_default_param ──────────────────────────────────────────────────────

    def _find_operator_non_default_param(self, tree: ast.AST) -> list[LintHit]:
        hits: list[LintHit] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            op_name = self._call_op_name(node)
            if op_name is None or op_name not in self.operator_rules:
                continue
            param_categories = self.operator_rules[op_name]
            for kw in node.keywords:
                if kw.arg is not None and kw.arg in param_categories:
                    # Emit the database pattern's own name as the category, so the analyzer's
                    # lookup-by-name lands on this exact pattern's evidence (ADR-036).
                    category = param_categories[kw.arg]
                    hits.append(
                        LintHit(
                            category,
                            node.lineno,
                            node.col_offset,
                            f"`{op_name}` called with watched param `{kw.arg}` "
                            f"({category}, Li et al. §3.2.2)",
                        )
                    )
                    break
        return hits

    @staticmethod
    def _call_op_name(call: ast.Call) -> str | None:
        # `some_op(...)` -> "some_op"; `t.some_op(...)` -> "some_op".
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None


# ── suppression (line / function / file level) ──────────────────────────────────────────
#
# The line and file markers are comments, which `ast` discards, so they are scanned from the source
# text. `file-ignore` is matched first (and on its own line), so it is never mistaken for the
# `ignore` marker, whose pattern requires `ignore` to follow `compile-lint:` directly.

#: `# compile-lint: ignore[pattern, …]` — suppress these patterns on this line.
_LINE_IGNORE = re.compile(r"#\s*compile-lint:\s*ignore\[([^\]]+)\]")
#: `# compile-lint: file-ignore[pattern, …]` — suppress these patterns in the whole file.
_FILE_IGNORE = re.compile(r"#\s*compile-lint:\s*file-ignore\[([^\]]+)\]")


def _split_patterns(group: str) -> set[str]:
    return {p.strip() for p in group.split(",") if p.strip()}


def _apply_suppressions(hits: list[LintHit], source: str, tree: ast.AST) -> list[LintHit]:
    """Drop hits silenced by a line / file comment marker or a function decorator."""
    file_ignored: set[str] = set()
    line_ignored: dict[int, set[str]] = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        if match := _FILE_IGNORE.search(line):
            file_ignored |= _split_patterns(match.group(1))
        elif match := _LINE_IGNORE.search(line):
            line_ignored.setdefault(lineno, set()).update(_split_patterns(match.group(1)))

    func_ignored = _collect_function_ignores(tree)

    def suppressed(hit: LintHit) -> bool:
        if hit.pattern_name in file_ignored:
            return True
        if hit.pattern_name in line_ignored.get(hit.line, set()):
            return True
        return any(
            start <= hit.line <= end and hit.pattern_name in patterns
            for start, end, patterns in func_ignored
        )

    return [hit for hit in hits if not suppressed(hit)]


def _collect_function_ignores(tree: ast.AST) -> list[tuple[int, int, set[str]]]:
    """`(start, end, patterns)` for each function decorated with `@compile_lint_ignore(...)`."""
    out: list[tuple[int, int, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            patterns = _decorator_patterns(node)
            if patterns:
                out.append((node.lineno, node.end_lineno or node.lineno, patterns))
    return out


def _decorator_patterns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    patterns: set[str] = set()
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and _callee_name(decorator.func) == "compile_lint_ignore"
        ):
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    patterns.add(arg.value)
    return patterns


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None
