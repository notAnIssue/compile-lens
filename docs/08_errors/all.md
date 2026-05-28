# compile-lens error codes

Stable registry of every `CLS-EXXXX` code compile-lens may surface. Codes are declared in
[`crates/cls-errors/src/lib.rs`](../../crates/cls-errors/src/lib.rs); the top-five
user-facing codes have a dedicated page below.

| Code | Variant | Page | Summary |
|---|---|---|---|
| `CLS-E0001` | `IoError` | [cls-e0001.md](cls-e0001.md) | File read failed (missing path / permission denied / etc.) |
| `CLS-E0002` | `SchemaParseError` | [cls-e0002.md](cls-e0002.md) | `.cls.json` failed to parse |
| `CLS-E0003` | `SchemaVersionMismatch` | — | Artifact's schema version is not the one this analyzer expects |
| `CLS-E0004` | `InvalidCliArgs` | — | CLI argument validation failed |
| `CLS-E0005` | `CollectorFailure` | — | A Python collector raised; the wrapped cause carries the detail |
| `CLS-E0006` | `AnalyzerInternalError` | — | Internal bug in an analyzer; please file a report |
| `CLS-E0007` | `MigrationRequired` | [cls-e0007.md](cls-e0007.md) | Older artifact needs `cl migrate` first |
| `CLS-E0008` | `SensitivePathDetected` | [cls-e0008.md](cls-e0008.md) | Path under `~/.ssh` / `~/.aws` / `~/.gnupg` refused on principle |
| `CLS-E0009` | `RedactionPolicyDemotionRefused` | [cls-e0009.md](cls-e0009.md) | Redaction is one-way; re-collect to recover raw data |
| `CLS-E0010` | `WorkingDirectoryAllowlistViolation` | — | MCP sandbox: requested path outside the allowlist |

The canonical docs live in this directory. URLs in rendered errors are **deferred until a
docs site exists** (see [ADR-022](../02_design_decisions/adr-022-error-handling.md));
errors today render the code (e.g. `CLS-E0001`) and the hint — look the code up here.
The five linked pages get a full write-up; the rest are auto-listed above with a one-line
summary until they earn one.
