# Tool 2a — `compile-diff`

> **Structurally diffs two compiled graphs.** Given a model compiled before and after a change,
> it reports which graph nodes were **added**, **removed**, or **modified**, by matching nodes
> across the two compiles on their structure rather than their names — node ids are not stable
> between compiles, so a name-based diff is useless. Each match carries a confidence, and the
> result carries two quality numbers (coverage and ambiguity).

When you change model code, the compiled graph changes too, but most of it stays the same. The
question `compile-diff` answers is *what actually changed in the graph* — not a textual diff of two
dumps (which is dominated by renamed temporaries), but a node-level **added / removed / modified**
classification you can act on.

---

## Part A — Theory & reference

### What it does

The input is two captured graphs (`base` and `head`), each the node-level FX structure inside a
`.cls.json` (the `compiled_graphs[].nodes[]` array — see [ADR-024](../02_design_decisions/adr-024-fx-graph-node-contract.md)).
Each node has an `id`, an `op_type` (e.g. `aten.sub.Tensor`), an **ordered** list of `inputs`
(upstream node ids), and free-form `attrs`. The output is an `IrGraphDiff`:

```
IrGraphDiff {
    added:    [head ids with no base counterpart],
    removed:  [base ids with no head counterpart],
    modified: [(base, head) pairs that matched but whose content changed],
    matched:  [(base, head, confidence) for every matched pair],
    match_coverage:          fraction of base nodes that found a match,
    anchor_uniqueness_ratio: fraction of base signatures owned by exactly one node,
}
```

### The algorithm: three phases

Matching is not exact graph isomorphism (NP-complete and unnecessary here — the two graphs are
near-aligned, differing by a small edit). It is a Weisfeiler–Lehman-style **structural signature**
plus a bounded expansion, in three phases.

**Phase 1 — Signature & anchoring.** Each node gets a signature: a deterministic hash of
`(op_type, the signatures of its inputs, the sorted op_types of its consumers)`. Two properties are
deliberate:

- Inputs are **sorted** for a *commutative* op (so `add(a, b)` and `add(b, a)` hash the same) but
  **kept in order** for a *non-commutative* op (so `sub(a, b)` differs from `sub(b, a)`). Which ops
  are commutative is a small, conservative whitelist — see
  [ADR-025](../02_design_decisions/adr-025-commutativity-whitelist.md).
- No node *name* enters the signature, so the diff is **rename-invariant**: renaming every node
  changes nothing.

A graph input (placeholder) is seeded by its **argument position** rather than its inputs/consumers —
the first input is a different thing from the second, and that identity must survive a downstream
restructuring. A signature that occurs exactly once in *each* graph, equal on both sides, is an
**anchor**: a high-confidence seed match.

```
sig(n) = hash(n.op_type,
              sorted(sig(i) for i in n.inputs) if commutative(n.op_type)
                                               else [sig(i) for i in n.inputs],
              sorted(c.op_type for c in consumers(n)))
anchors = { (b, h) : sig(b) unique in base, sig(h) unique in head, sig(b) == sig(h) }
```

**Phase 2 — Neighborhood expansion.** Most nodes are not unique enough to anchor alone, but sit next
to ones that are. Breadth-first from each anchor over both graphs at once, up to `d_max = 3` hops, a
neighbor is matched to its counterpart when they share an `op_type` and line up by edge position
(the k-th input to the k-th input; a consumer to a consumer fed at the same position). Matching is
greedy (first valid wins) and one-to-one. `d_max` is a hyperparameter, not a correctness bound: a
pair matched at depth 3 is not expanded further.

**Phase 3 — Residual classification.** From the final matching:

- a base node with no counterpart is **removed**, a head node with no counterpart is **added**;
- a matched pair whose `attrs` differ is **modified** — the signature ignores attrs, so an
  attribute-only change (`add(a, b, alpha=1)` → `alpha=2`) still matches structurally and is caught
  by comparing the pair's attrs;
- a non-commutative operand swap (`sub(a, b)` → `sub(b, a)`) makes the node's signature differ so it
  doesn't anchor, and position-aligned expansion can't realign it; residual **recovers** it — two
  unmatched nodes with the same op_type and the same *set* of matched inputs are the same node with
  reordered operands, classified `modified`.

### Complexity

`O(Σ_v deg(v) · log deg(v) + V · d_max)` — the `deg·log deg` term is sorting commutative inputs while
hashing; `V · d_max` is the bounded expansion. Near-linear when fan-in is small, which it is for FX
graphs.

### Confidence contract

The matching is approximate, so the result is self-describing rather than claimed exact:

- **`match_coverage`** — fraction of base nodes matched. Low coverage means the change was large (or
  the matcher struggled); treat the diff with more suspicion.
- **`anchor_uniqueness_ratio`** — fraction of the base graph's signatures that belong to exactly one
  node. Near 1.0 means almost every node is structurally unique (anchoring is reliable); lower means
  many look-alikes.
- **per-match `confidence`** in `[0, 1]` — `0.5 · uniqueness + 0.5 · neighborhood_agreement`, where
  uniqueness is `1 / signature-bucket-size` and agreement is the fraction of a node's neighbors whose
  match is the counterpart's neighbor.

These are gated in CI: the matcher must clear a median coverage of 0.70 on real model graphs and a
gold-corpus precision of 0.90 (the **Algorithmic invariants** release-blocker job).

---

## Part B — Examples

### Producing the inputs

Capture the two graphs with the Python collector (`CompileArtifactCollector`), which hooks a
`torch.compile` and writes the aten-normalized FX graph into a `.cls.json`:

```python
from compile_lens.collectors.artifact import CompileArtifactCollector

# before your change
c = CompileArtifactCollector("base.cls.json")
c.capture(model_before, example_input)
c.finalize()

# after your change
c = CompileArtifactCollector("head.cls.json")
c.capture(model_after, example_input)
c.finalize()
```

### The diff

```bash
cl diff --base before.cls.json --head after.cls.json --format markdown
```

For a non-commutative operand swap — `head` computes `sub(b, a)` where `base` computed `sub(a, b)`:

```markdown
## Compile Diff

0 added · 0 removed · 1 modified · 3 matched

### Modified
- `n2`

_match coverage 1.00 · anchor uniqueness 1.00_
```

`--format json` emits the whole `IrGraphDiff` (the machine-readable contract — `added` / `removed` /
`modified`, the `matched` triples with their confidence, `match_coverage`, `anchor_uniqueness_ratio`);
`--format text` is the same summary in plain console output. The matcher is also a pure library
function — `cls_wl_diff::diff_graphs(before, after) -> IrGraphDiff`, deterministic and torch-free — if
you want to embed it rather than shell out.

> A Python binding for the diff is not yet shipped; today the surfaces are the `cl diff` command and
> the Rust library function above.

### GitHub Action

Gate a PR on the compile diff by writing the Markdown into the job summary:

```yaml
# .github/workflows/compile-diff.yml
- run: cl diff --base baseline.cls.json --head pr.cls.json --format markdown >> "$GITHUB_STEP_SUMMARY"
```

---

## Part C — Limitations & failure cases

### Variable rename — handled

A pure rename (every node renamed, structure identical) is **matched correctly**: the signature uses
no node names, so renaming changes nothing. (A naive WL implementation that let SSA names leak into
consumer signatures would misclassify this as add+remove; this implementation does not.)

### Operand swap on a non-commutative op — detected

`sub(a, b)` → `sub(b, a)` is reported as `modified`, not silently dropped. This is the most important
guarantee — a missed non-commutative swap is a silent false negative the numerics would change while
the tool says nothing — and it is enforced by a release-blocker regression test.

### Input-change ripples — can over-report

When a node's *input op* changes, that node's signature changes, so it may not match and will be
reported as add+remove even if the node itself is unchanged. Two real cases: a **fusion** that splits
a fused op (the shared inputs' consumers change), and **dropping a linear's bias** (which turns
`addmm` into `mm`, so the following `relu` reads a different op). The matcher reports a change where a
human might say "same node, different neighbor" — a false positive that lowers coverage but never
silently hides a real change.

### Ambiguous repeated structure — lower confidence

A graph with many structurally identical nodes (several identical layers) has a low
`anchor_uniqueness_ratio`: fewer unique signatures to anchor on, so expansion does more of the work
and matches are less certain. The confidence numbers surface this rather than hiding it.

---

## Related

- [ADR-024 — FX-graph node contract](../02_design_decisions/adr-024-fx-graph-node-contract.md): the `nodes[]` structure this consumes.
- [ADR-025 — commutativity whitelist](../02_design_decisions/adr-025-commutativity-whitelist.md): which ops sort their operands.
- [Tool 1 — `recompile-summary`](recompile_summary.md): the other analyzer over the same `.cls.json`.
