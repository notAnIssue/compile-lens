"""Tests for the Layer-1 AST lint scanner (Tool 4).

The scanner is pure static analysis: it parses the user's source with ``ast`` and **never executes
it**. These cover the two v0 patterns (positive + negative) and the no-execution guarantee.
"""

from compile_lens.collectors.lint import LintPatternScanner


def _names(hits) -> list[str]:
    return sorted({h.pattern_name for h in hits})


# ── in_place_op_on_alias (Li et al. §3.2.3) ─────────────────────────────────────────────


def test_in_place_subscript_on_alias_is_flagged() -> None:
    # `expand` returns a view; writing through it in place is the anti-pattern.
    src = "y = x.expand(2, 3)\ny[0] = 1\n"
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(src))


def test_in_place_method_on_alias_is_flagged() -> None:
    # A trailing-underscore method (`add_`) is in-place; `transpose` aliased `y`.
    src = "y = x.transpose(0, 1)\ny.add_(1)\n"
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(src))


def test_alias_chain_is_tracked() -> None:
    # z aliases y aliases x; the in-place write on z still counts.
    src = "y = x.expand(2, 3)\nz = y.view(6)\nz[0] = 1\n"
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(src))


def test_clone_then_in_place_is_not_flagged() -> None:
    # `clone` copies — y is not an alias, so the in-place write is safe.
    src = "y = x.clone()\ny[0] = 1\n"
    assert "in_place_op_on_alias" not in _names(LintPatternScanner().scan(src))


def test_in_place_on_fresh_tensor_is_not_flagged() -> None:
    # y is freshly constructed, not aliased from anything.
    src = "y = torch.zeros(3)\ny[0] = 1\n"
    assert "in_place_op_on_alias" not in _names(LintPatternScanner().scan(src))


def test_dunder_method_is_not_treated_as_in_place() -> None:
    # A dunder like `__len__` ends in `_` but is not an in-place op; must not false-positive.
    src = "y = x.expand(2, 3)\ny.__len__()\n"
    assert "in_place_op_on_alias" not in _names(LintPatternScanner().scan(src))


# ── operator-family patterns: granular categories (Li et al. §3.2.2, ADR-036) ───────────
#
# operator_rules maps operator -> {param: pattern_category}; the scanner emits *that category*
# (the database pattern's own name) so the analyzer can look the exact pattern up.

_RULES = {"some_op": {"flag": "some_op_flag_bug"}}


def test_watched_op_with_the_param_emits_the_patterns_category() -> None:
    scanner = LintPatternScanner(operator_rules=_RULES)
    # The emitted category is the rule's pattern name, not a generic label.
    assert "some_op_flag_bug" in _names(scanner.scan("some_op(x, flag=True)\n"))


def test_watched_op_without_the_param_is_not_flagged() -> None:
    scanner = LintPatternScanner(operator_rules=_RULES)
    assert "some_op_flag_bug" not in _names(scanner.scan("some_op(x)\n"))


def test_method_form_of_watched_op_is_flagged() -> None:
    scanner = LintPatternScanner(operator_rules=_RULES)
    assert "some_op_flag_bug" in _names(scanner.scan("t.some_op(flag=2)\n"))


def test_two_operator_rules_emit_distinct_categories() -> None:
    # The fan-out: different (operator, param) rules produce different categories, so each
    # operator-family bug maps to its own database entry.
    scanner = LintPatternScanner(operator_rules={"op_a": {"p": "cat_a"}, "op_b": {"q": "cat_b"}})
    found = _names(scanner.scan("op_a(x, p=1)\nop_b(y, q=2)\n"))
    assert "cat_a" in found and "cat_b" in found


# ── the no-execution guarantee + finding shape ──────────────────────────────────────────


def test_scanner_never_executes_user_code() -> None:
    # If the scanner exec'd this, it would raise / shell out. A pure AST parse must be inert.
    src = "raise RuntimeError('boom')\nimport os\nos.system('echo hacked')\n"
    hits = LintPatternScanner().scan(src)  # must not raise
    assert isinstance(hits, list)


def test_hit_carries_source_location() -> None:
    src = "y = x.expand(2, 3)\ny[0] = 1\n"
    hits = [h for h in LintPatternScanner().scan(src) if h.pattern_name == "in_place_op_on_alias"]
    assert hits and hits[0].line == 2
