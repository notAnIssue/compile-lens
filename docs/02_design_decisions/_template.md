# ADR-NNN: <short title>

- **Status**: Proposed | Accepted | Superseded by ADR-XXX
- **Date**: YYYY-MM-DD
- **Deciders**: <name(s)>
- **Related**: <other ADRs / in-repo design docs / schema files>

## Context

What problem forces a decision? What constraints apply (cross-language contract,
SemVer, security, performance)? Why is the cost of getting this wrong high?

State the framing explicitly, and call out any *pseudo-criteria* that look relevant
but are not (e.g. a "performance" axis where nothing is on a hot path).

## Decision

The decision, stated as a rule that future contributors can apply without re-litigating.
One paragraph; bullet the rule if it has parts.

## Consequences

- **Positive**: what this buys us.
- **Negative / costs**: what this makes harder, and the discipline it imposes.
- **Follow-ups**: anything that must be true elsewhere for this to hold.

## Alternatives considered

List the options. Then a **weighted decision matrix** — mandatory for any ADR-level
decision:

1. Define criteria, assign weights summing to **10**, justify each weight.
2. Score **every** option 0–10 on each criterion (same scale — not a justification
   of the pre-chosen answer).
3. Weighted contribution = `weight × (raw/10)`; sum per option.
4. Readout + which weight/score would flip the result.

| Criterion (weight) | Option A | Option B | Option C |
|---|---|---|---|
| ... (w) | r (c) | r (c) | r (c) |
| **Weighted total / 10** | ... | ... | ... |
