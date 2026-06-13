"""Validate Tool 4 against the taxonomy paper's own example bug listings, and guard false positives.

The headline acceptance: run the scanner (configured from the production correctness database) over
the paper's Listings 1-5 and assert it detects **exactly** the ones inside its documented coverage —
the operator-non-default-param case (Listing 3, diag_embed) and the in-place-on-alias case (Listing
5) — while staying silent on the ones a static AST scan cannot reach (Listing 1 graph-semantic,
Listing 2 graph-caching, Listing 4 a single-operator numerical bug not in the catalogue).

This is the empirical complement to the coverage section of ``docs/03_tools/compile_lint.md``: it
proves the boundary in both directions — what the linter catches *and* what it correctly declines to
flag. A lint that over-claims is worse than none, so "stays silent on the out-of-reach listings" is
as much the acceptance as "detects the in-reach ones."
"""

from __future__ import annotations

from pathlib import Path

from compile_lens.collectors.lint import LintPatternScanner
from compile_lens.correctness_db import load_operator_rules

REPO_ROOT = Path(__file__).resolve().parents[2]
_RULES = load_operator_rules(REPO_ROOT / "data" / "correctness_db.toml")


def _categories(source: str) -> set[str]:
    return {hit.pattern_name for hit in LintPatternScanner(_RULES).scan(source)}


# ── the paper's Listings 1-5 (faithful trigger snippets; parsed, never executed) ─────────

# Listing 1 — Graph Semantic Capturing: a TorchDispatchMode rewrites add->mul. Out of reach: the
# bug is in how compile captures execution context, not visible as a source anti-pattern.
LISTING_1 = """
class RewriteAddToMul(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.add.Tensor:
            func = torch.ops.aten.mul.Tensor
        return func(*args, **(kwargs or {}))

def model(x):
    return x + x
"""

# Listing 2 — Graph Caching: internal state mutates across repeated forwards. Out of reach: only
# surfaces under repeated execution, not in a single static scan.
LISTING_2 = """
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = -1
        self.cache = torch.tensor([2, 3, 4, 5, 6, 7])

    def forward(self):
        self.value += 1
        return self.cache[self.value]
"""

# Listing 3 — Operator Transformation: diag_embed with a non-default (negative) dim. IN COVERAGE
# (operator-family) -> diag_embed_nondefault_dim.
LISTING_3 = """
@torch.compile
def model(x):
    return torch.diag_embed(input=x, dim1=-1, dim2=0, offset=1)
"""

# Listing 4 — Low-level codegen: polygamma at the n==1 boundary. A detectable *class* (single
# operator) but not catalogued — its `n` is conventionally positional, so the keyword detector can't
# ground a clean fixture; the scanner is therefore silent here.
LISTING_4 = """
@torch.compile
def model(x):
    return torch.special.polygamma(x, n=1)
"""

# Listing 5 — In-Place Operation Handling: in-place write through an `expand` alias. IN COVERAGE
# (structural) -> in_place_op_on_alias.
LISTING_5 = """
@torch.compile
def model(x):
    y = x.expand(2, *x.shape)
    y[0, 0] = 5
    return y
"""


def test_listing_3_operator_param_is_detected() -> None:
    assert "diag_embed_nondefault_dim" in _categories(LISTING_3)


def test_listing_5_in_place_on_alias_is_detected() -> None:
    assert "in_place_op_on_alias" in _categories(LISTING_5)


def test_out_of_reach_listings_are_not_flagged() -> None:
    # Listings 1, 2, 4 are outside the static AST detector's coverage; flagging any of them would be
    # a false positive that overstates the tool's reach.
    assert _categories(LISTING_1) == set()
    assert _categories(LISTING_2) == set()
    assert _categories(LISTING_4) == set()


def test_coverage_boundary_is_exactly_the_two_in_reach_listings() -> None:
    # The full boundary in one assertion: detect the two in-reach, silent on the other three.
    detected = {
        name
        for name, src in [
            ("L1", LISTING_1),
            ("L2", LISTING_2),
            ("L3", LISTING_3),
            ("L4", LISTING_4),
            ("L5", LISTING_5),
        ]
        if _categories(src)
    }
    assert detected == {"L3", "L5"}


# ── false-positive guard: idiomatic / near-miss code must stay silent ─────────────────────
#
# Each snippet is structurally close to a pattern but safe; none should produce a finding. This
# guards the coarse Layer-1 detectors from creeping into common, correct code.
FP_SAFE = [
    "y = x.clone()\ny[0] = 1",  # clone shares no storage — not an alias
    "y = x.expand(2, 3)\nz = y.sum()",  # alias created but never written in place
    "out = torch.diag_embed(x)",  # watched op, but no watched parameter passed
    "out = F.interpolate(x, scale_factor=2)",  # default mode
    "out = F.pad(x, (1, 1))",  # default (constant) padding
    "out = torch.repeat_interleave(x, repeats)",  # no output_size
    "x.index_put_((idx,), vals)",  # no accumulate
    'out = some_other_op(x, mode="bicubic")',  # mode keyword, but on an unwatched operator
]


def test_idiomatic_code_does_not_false_positive() -> None:
    for snippet in FP_SAFE:
        assert _categories(snippet) == set(), f"false positive on:\n{snippet}"
