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
    """Scan Python source for Tool 4's v0 anti-patterns, purely via the AST.

    ``watched_ops`` maps an operator name to the set of parameter names whose non-default use is
    flagged as ``operator_non_default_param``. It defaults to empty: with no watch list that pattern
    matches nothing (the real list is supplied by the correctness database), while
    ``in_place_op_on_alias`` is structural and always active.
    """

    def __init__(self, watched_ops: dict[str, set[str]] | None = None) -> None:
        self.watched_ops: dict[str, set[str]] = dict(watched_ops) if watched_ops is not None else {}

    def scan(self, source: str, filename: str = "<unknown>") -> list[LintHit]:
        # Parse only — the user's code is never compiled-and-run, so scanning is side-effect free.
        tree = ast.parse(source, filename=filename)
        aliases = self._collect_aliases(tree)
        hits: list[LintHit] = []
        hits.extend(self._find_in_place_on_alias(tree, aliases))
        hits.extend(self._find_operator_non_default_param(tree))
        return hits

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
            if op_name is None or op_name not in self.watched_ops:
                continue
            watched_params = self.watched_ops[op_name]
            for kw in node.keywords:
                if kw.arg is not None and kw.arg in watched_params:
                    hits.append(
                        LintHit(
                            "operator_non_default_param",
                            node.lineno,
                            node.col_offset,
                            f"`{op_name}` called with non-default `{kw.arg}` (Li et al. §3.2.2)",
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
