"""Tests for Tool 4 suppression: line / function / file level.

The line and file markers are comments — which ``ast`` discards — so they are scanned from the
source text; the function-level decorator is read from the AST. Still pure static analysis.
"""

from compile_lens.collectors.lint import LintPatternScanner


def _names(hits) -> set[str]:
    return {h.pattern_name for h in hits}


# An unsuppressed in_place_op_on_alias: `expand` aliases, the subscript write is in-place.
BASE = "y = x.expand(2, 3)\ny[0] = 1\n"


def test_baseline_is_flagged() -> None:
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(BASE))


def test_line_level_ignore_suppresses() -> None:
    src = "y = x.expand(2, 3)\ny[0] = 1  # compile-lint: ignore[in_place_op_on_alias]\n"
    assert "in_place_op_on_alias" not in _names(LintPatternScanner().scan(src))


def test_line_level_ignore_of_another_pattern_does_not_suppress() -> None:
    src = "y = x.expand(2, 3)\ny[0] = 1  # compile-lint: ignore[some_other_pattern]\n"
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(src))


def test_line_level_ignore_only_suppresses_its_own_line() -> None:
    # The ignore on line 2 must not silence the in-place op on line 3.
    src = "a = x.expand(2, 3)\na[0] = 1  # compile-lint: ignore[in_place_op_on_alias]\na[1] = 2\n"
    hits = [h for h in LintPatternScanner().scan(src) if h.pattern_name == "in_place_op_on_alias"]
    assert [h.line for h in hits] == [3]


def test_file_level_ignore_suppresses_the_whole_file() -> None:
    src = "# compile-lint: file-ignore[in_place_op_on_alias]\ny = x.expand(2, 3)\ny[0] = 1\n"
    assert "in_place_op_on_alias" not in _names(LintPatternScanner().scan(src))


def test_file_ignore_is_not_misread_as_a_line_ignore() -> None:
    # `file-ignore[other]` must not be parsed as `ignore[other]` (it contains the substring),
    # and it names only `other`, so the real pattern stays flagged.
    src = "# compile-lint: file-ignore[some_other_pattern]\ny = x.expand(2, 3)\ny[0] = 1\n"
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(src))


def test_function_level_decorator_suppresses_within() -> None:
    src = (
        '@compile_lint_ignore("in_place_op_on_alias")\n'
        "def f(x):\n"
        "    y = x.expand(2, 3)\n"
        "    y[0] = 1\n"
    )
    assert "in_place_op_on_alias" not in _names(LintPatternScanner().scan(src))


def test_function_level_decorator_does_not_suppress_outside() -> None:
    src = (
        '@compile_lint_ignore("in_place_op_on_alias")\n'
        "def f(x):\n"
        "    pass\n"
        "y = x.expand(2, 3)\n"
        "y[0] = 1\n"
    )
    assert "in_place_op_on_alias" in _names(LintPatternScanner().scan(src))
