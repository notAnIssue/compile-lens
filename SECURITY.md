# Security Policy

`compile-lens` takes security seriously. This document explains how to report a
vulnerability and what to expect in return.

---

## Reporting a vulnerability

**Do not** open a public GitHub issue for a security vulnerability.

Please use one of the following channels:

1. **GitHub Security Advisory (preferred)**: <https://github.com/notAnIssue/compile-lens/security/advisories/new>
2. **Email**: `security@compile-lens.dev` (PGP key fingerprint published at <https://compile-lens.dev/security/pgp.txt>)

Please include:

- Affected version (or the `main` HEAD commit SHA)
- Reproduction steps / proof-of-concept
- Impact assessment (what an attacker can do)
- A suggested fix (optional, but welcome)
- Whether you plan to disclose publicly, and if so the intended timeline

---

## Disclosure timeline

| Stage | SLA |
|---|---|
| **Acknowledge receipt** | within 7 days |
| **Initial severity assessment** | within 14 days |
| **Coordinated disclosure window** | 90 days from initial assessment (extendable by mutual agreement) |
| **Patch release + public advisory** | as soon as a fix is ready, no later than the end of the disclosure window |

If a vulnerability is being actively exploited in the wild, we may release a patch before
the 90-day window ends.

---

## Scope

### In scope

The following classes of vulnerability are covered by this policy:

- **IP leakage**: an artifact (`.cls.json` / `report.html`) under the default redaction
  policy still leaks sensitive information it claims to remove (paths, secrets, source,
  kernel internals)
- **XSS / script injection** in `report.html` via op name / file path / error message /
  any other user-controlled input
- **MCP server sandbox escape**: an agent connected over MCP can read files outside the
  allowlist, write outside the designated area, or run commands outside the tool allowlist
- **Path traversal**: in any CLI that accepts a file path (`cl collect`, `cl diff`, etc.)
- **Schema-migration data leak**: redaction is bypassed when an old artifact is upgraded to
  the current schema (e.g. a legacy artifact upgraded without applying default redaction)
- **Denial of service**: a crafted `.cls.json` that makes the analyzer consume unbounded
  memory / CPU
- **Insecure defaults**: a documented default behavior with a security implication not
  disclosed to the user

### Out of scope (generally not fixed)

- Issues in PyTorch / Triton / CUDA themselves (please report upstream)
- Issues requiring root / admin host access (already trusted)
- Side-channel attacks against the user's ML workload (outside the tool's remit)
- Social engineering / phishing aimed at users
- Issues that appear only under non-default configuration, unless that configuration is
  widely used (file as an enhancement)
- Missing security headers on a third-party-hosted docs site (separate from the tool itself)

---

## Hall of fame

Confirmed valid reports are credited in the project's `THANKS.md` (with the reporter's
consent) and in the published advisory.

---

## Coordinated disclosure best practices

We commit to the following:

1. We will not publicly identify a reporter without their consent.
2. We will share the advisory draft with the reporter before publication for accuracy review.
3. We will not sue, threaten, or take legal action against a security researcher acting in
   good faith within the scope and timeline of this policy.
4. We do not require a reporter to sign an NDA as a condition of reporting (though we
   appreciate confidentiality during the disclosure window).

---

## What this project does **not** promise (transparency)

`compile-lens` is an OSS project maintained by 1–2 people. We do not have:

- 24/7 on-call security response
- SOC 2 / ISO 27001 / FedRAMP certification
- Formal supply-chain attestation (Sigstore signing, SLSA Build Level 3 provenance) **in the
  early stage** — these are post-adoption candidates (see `docs/06_security/threat_model.md`
  §4, "Out-of-scope security areas")
- Formal penetration-testing engagements

If your use case requires any of the above, assess whether `compile-lens` meets your needs
before deploying it in a security-sensitive production environment.

---

## Related documents

- `docs/06_security/threat_model.md` — full STRIDE-style threat analysis
- `docs/06_security/redaction_policy.md` — what each redaction policy level scrubs
- `docs/06_security/sandbox_design.md` — MCP server sandbox spec (Phase 2, not yet shipped)
- `docs/02_design_decisions/` — security/privacy-related architecture decisions (ADRs)

---

## Versioning of this policy

This policy is versioned alongside the project's SemVer. Substantive changes (e.g. a new
in-scope class) are noted in `CHANGELOG.md`. The current policy applies to all currently
supported releases (latest minor + N-1).
