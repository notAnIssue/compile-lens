# ADR-025: Hardcoded, conservative commutativity whitelist for operand-order diffing

- **Status**: Accepted
- **Date**: 2026-06-10
- **Deciders**: project maintainer
- **Related**: ADR-024 (FX-graph node contract — the ordered `inputs` this policy interprets); the WL-signature diff (the diff is a pure function with no torch dependency).

## Context

The WL-signature diff (Tool 2a) decides, for each graph node, whether reordering its
operands is a real change. For a **commutative** operation it sorts the operand
signatures, so `add(a, b)` and `add(b, a)` hash identically and a reorder stays silent.
For a **non-commutative** operation it preserves operand order, so `sub(a, b)` and
`sub(b, a)` hash differently and a reorder is reported. Getting this wrong is not a cosmetic
error: marking a non-commutative op as commutative would sort its operands and **silently
hide a real change** — a class of bug a user could not easily detect, since the numerics
change while the tool reports "no difference."

So the question is where the commutativity knowledge comes from. Two facts frame it:

1. The diff is a **pure function with no PyTorch dependency** (D4). It runs over `.cls.json`
   artifacts, long after capture, in Rust. It cannot call into torch at diff time.
2. aten exposes **no reliable per-op `is_commutative` attribute** to query. Commutativity is
   not a first-class property in the dispatcher, so any approach hand-maintains the list.

A *pseudo-criterion* to name and set aside: runtime **extensibility** (letting a deployment
override commutativity per-op). It looks relevant, but there is no consumer for it today —
the conservative default already handles unknown ops safely, and the in-code
`CommutativitySet` type can already carry an alternate set for tests. Weighting it heavily
would distort the decision toward speculative configurability.

## Decision

Maintain a small, **hardcoded** set of commutative operation names in Rust
(`cls-wl-diff/src/commutativity.rs`), with a **conservative default: an op is
non-commutative unless explicitly listed.**

The default makes the failure modes asymmetric, which is the whole point:

- Wrongly *omitting* a commutative op makes a pure reorder look like a modification —
  noisy, but safe and visible.
- Wrongly *including* a non-commutative op sorts its operands and hides a real change — a
  silent miss.

Therefore the list contains only operations whose commutativity is unambiguous
(`add`, `mul`, `bitwise_and`/`or`/`xor`, `logical_and`/`or`, `minimum`, `maximum`, `eq`,
`ne`), and everything unknown falls through to non-commutative, which can only over-report.
Op-type strings are normalized (namespace and overload suffix stripped) before lookup, so
`add`, `aten.add`, and `torch.ops.aten.add.Tensor` all resolve to `add`.

## Consequences

- **Positive**: the diff stays a self-contained pure function — no torch at diff time, no
  per-node commutativity flag bloating the artifact, one source of truth. The conservative
  default guarantees that any list error can only produce visible noise, never a silent miss.
- **Negative / costs**: the list is hand-maintained, and a genuinely commutative op left off
  the list yields false-positive "modified" reports until someone adds it. This is the
  intended trade (visible over-report over silent miss).
- **Follow-ups**: the WL-signature computation takes a `&CommutativitySet` so the policy is
  injected rather than baked into the algorithm; `CommutativitySet::standard()` is the
  production set.

## Alternatives considered

- **A — Hardcoded Rust set, conservative default** (chosen).
- **B — Capture-time annotation**: the Python collector (which has torch) marks each node's
  commutativity and stores it per-node in the artifact for the Rust diff to read.
- **C — Hybrid**: a hardcoded core set plus a user-overridable config / per-node override.

Weighted decision matrix (weights sum to 10; every option scored 0–10 on the same scale,
weighted contribution in parentheses):

| Criterion (weight) | A hardcoded | B capture-time | C hybrid |
|---|---|---|---|
| Architecture fit — pure-Rust diff, no torch at diff time (3) | 9 (2.70) | 4 (1.20) | 6 (1.80) |
| Correctness safety — conservative, asymmetric failure (3) | 9 (2.70) | 7 (2.10) | 8 (2.40) |
| Maintainability — single source, no second list (2) | 8 (1.60) | 4 (0.80) | 5 (1.00) |
| Artifact self-containment / size (1) | 9 (0.90) | 5 (0.50) | 7 (0.70) |
| Extensibility — override / custom sets (1) | 6 (0.60) | 6 (0.60) | 8 (0.80) |
| **Weighted total / 10** | **8.50** | **5.20** | **6.70** |

- **Weight justification**: architecture fit and correctness safety are heaviest (3 each) —
  the first because D4 (pure-Rust diff) is a load-bearing design choice that B fights, the
  second because a wrong call here is a silent correctness bug. Maintainability (2): a second
  list is a recurring drift risk. Self-containment (1) and extensibility (1) are real but
  minor; extensibility is the named pseudo-criterion with no current consumer.
- **Readout**: A wins at 8.50, dominating on both heavy axes.
- **B's fatal flaw**: the diff has no torch at diff time, so B forces commutativity into the
  Python collector and a per-node flag in every artifact — coupling the data format to one
  concern and bloating it — yet, because aten has no reliable `is_commutative` attribute, the
  list is *still* hand-maintained. B pays the coupling cost for no reduction in manual work.
- **C's flaw**: it adds an override/config surface with no consumer today; the conservative
  default already makes unknown ops safe, so the override buys nothing now (YAGNI).
- **What would flip it**: if a future requirement needed per-deployment commutativity
  customization — for example custom ops whose commutativity varies by user — C's override
  would start to pay off. And if the diff ever moved into the capture process and gained torch
  access, B's "annotate at capture" could become cheaper. Neither holds now; the pure-function
  design (D4) is deliberate, so revisit only if it changes.
