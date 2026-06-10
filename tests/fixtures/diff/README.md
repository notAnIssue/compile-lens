# Compile-diff fixture corpus

Controlled input pairs the Tool 2a (*compile-diff*) test suite consumes. Each fixture
is a `base` compile and a `head` compile of the same model, plus an oracle of the diff
the analyzer is expected to derive between them.

Unlike the Tool 1 recompile corpus — which is captured from real PyTorch sessions — the
pairs here are **synthetic and labelled as such**: small, hand-constructed FX graphs that
isolate one diff behaviour each. Synthetic is the right tool for correctness tests, because
it lets each fixture exercise exactly one phenomenon (an added node, an operand swap, a
rename) with no incidental noise. Real captured model pairs are a separate, later addition
that depends on the compile-artifact collector being able to emit node-level graphs; they
are intentionally **not** hand-faked here, because a fabricated graph posing as a real
capture would be misleading.

## Layout

Each scenario is a directory with three files:

- `base.cls.json` — the "before" compile, a valid `.cls.json` (schema v0.5.0). Its single
  `compiled_graphs[0].nodes[]` array carries the node-level FX structure (the `nodes` field
  added in ADR-024): each node has an `id`, an `op_type`, ordered `inputs` (upstream node
  ids), and optional `attrs`.
- `head.cls.json` — the "after" compile, same shape.
- `expected.diff.json` — the oracle (see below).

## Oracle format (`expected.diff.json`)

The oracle pins the **qualitative** classification the diff must produce, by node id:

| Field | Meaning |
|---|---|
| `description` | One line: what this pair changes. |
| `added` | Head node ids that have no counterpart in base. |
| `removed` | Base node ids that have no counterpart in head. |
| `modified` | Node ids present in both but changed (different attrs, or — for non-commutative ops — different operand order). |
| `matched` *(optional)* | Explicit `[base_id, head_id]` pairs, where naming them out matters (e.g. the rename case). |
| `known_limitation` *(optional)* | `true` marks a pair the algorithm is documented to get wrong; downstream tests treat it as an example, not a hard pass. |
| `note` | Why this case matters and what it guards against. |

It deliberately does **not** pin the algorithm's emergent scores (`match_coverage`,
`anchor_uniqueness_ratio`, per-match confidence). Those are outputs of the matching
algorithm, checked against thresholds in their own tests — not part of the per-fixture
spec, which would otherwise have to change every time the scoring is tuned.

## Inventory

| Fixture | What it exercises |
|---|---|
| `add_node` | Head appends exactly one node; nothing existing is rewired. Smallest add case. |
| `remove_node` | Mirror of `add_node`: one base node is absent from head. |
| `modify_attr` | Same `op_type` and operands, different `attrs` (`alpha` on `aten.add`). Must be `modified`, not add+remove. |
| `operand_swap_sub` | `aten.sub(a, b)` becomes `sub(b, a)`. Non-commutative, so **must** be detected as `modified`. |
| `operand_swap_div` | Same for `aten.div`. |
| `operand_swap_matmul` | Same for `aten.matmul`. |
| `commutative_add` | `aten.add(a, b)` becomes `add(b, a)`. Commutative, so the diff **must stay silent** (empty). |
| `variable_rename` | Every node id renamed, structure identical. Documented limitation: may be misclassified as add+remove. |
| `fusion_lost` | A fused `aten.addmm` in base is split into `aten.mm` + `aten.add` in head. |

The `operand_swap_*` and `commutative_add` fixtures are the two halves of the operand-order
guarantee: non-commutative swaps must be caught, commutative swaps must be ignored. Getting
either wrong is a silent correctness bug a user could not easily spot, which is why they are
the corpus's most important cases.
