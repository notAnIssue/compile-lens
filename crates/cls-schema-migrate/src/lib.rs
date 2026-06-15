//! `cls-schema-migrate` — schema version **detection** for `.cls.json` artifacts.
//!
//! # Pre-V1 design (current)
//!
//! The schema is still iterating with feature work. There is **no migration ladder** —
//! every per-version migration would be throwaway (the same developer changes schema +
//! collector + analyzer in one sitting; no external artifacts exist that need preserving).
//! This crate's only job, pre-V1, is to **detect** whether an incoming artifact is at
//! the current schema and **refuse** otherwise (`CLS-E0003 SchemaVersionMismatch`) —
//! protecting against silent drift between collaborators or future-you when fixtures /
//! examples are reused after a schema change.
//!
//! # V1 trigger (when the full ladder activates)
//!
//! Freeze as **V1** when:
//! - all Phase 0 + Phase 1 TODO features are implemented;
//! - the toolkit has been used / debugged enough that **no further feature work is
//!   expected to drive a schema change**.
//!
//! At that point ("public alpha"-style release) every subsequent schema bump ships a
//! real per-version migration function — the migration discipline (with the pre-V1
//! exception of detect-and-refuse described above).

use cls_errors::ClsError;
use cls_schema::ClsArtifact;
use std::path::Path;

/// Schema version this binary currently treats as "current".
///
/// Kept in sync with [`cls_schema::SCHEMA_VERSION`]; the two should never disagree.
pub const CURRENT_VERSION: &str = cls_schema::SCHEMA_VERSION;

/// Inspect just the top-level `schema_version` field of the file at `path` and verify
/// it matches [`CURRENT_VERSION`]. Returns the version string on success, or
/// [`ClsError::SchemaVersionMismatch`] when it does not (pre-V1: the user re-collects).
///
/// Cheap on purpose — does not deserialize the whole artifact, so it works even on
/// foreign artifacts the binary cannot model.
pub fn detect_schema_version(path: &Path) -> Result<String, ClsError> {
    let text = std::fs::read_to_string(path).map_err(|source| ClsError::IoError {
        path: path.display().to_string(),
        source,
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|source| ClsError::SchemaParseError { source })?;
    let version = value
        .get("schema_version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ClsError::SchemaVersionMismatch {
            found: "(missing)".into(),
            expected: CURRENT_VERSION.into(),
        })?
        .to_string();
    if version == CURRENT_VERSION {
        Ok(version)
    } else {
        Err(ClsError::SchemaVersionMismatch {
            found: version,
            expected: CURRENT_VERSION.into(),
        })
    }
}

/// "Migrate" `input_path` to the current schema, writing to `output_path`.
///
/// Pre-V1 the only allowed case is: input is **already at** the current version, in which
/// case the file is **byte-copied** (D10: same input must produce byte-identical output —
/// verifiable by `cmp`); any other version returns [`ClsError::SchemaVersionMismatch`] and
/// **no output file is written** so the caller can't be fooled by a half-written result.
/// Post-V1, this function will dispatch to the matching per-version transform.
pub fn migrate_to_current(input_path: &Path, output_path: &Path) -> Result<(), ClsError> {
    detect_schema_version(input_path)?;
    let bytes = std::fs::read(input_path).map_err(|source| ClsError::IoError {
        path: input_path.display().to_string(),
        source,
    })?;
    std::fs::write(output_path, bytes).map_err(|source| ClsError::IoError {
        path: output_path.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Read a `.cls.json` from disk, verify its schema version, and fully deserialize it
/// into a [`ClsArtifact`].
///
/// This is the **shared baseline-pairing seam** (ADR-033 §5): any analyzer that needs a
/// *parsed* artifact off disk — `cl recompile-summary` loading its session, the same
/// command loading a `--baseline` for the regression diff, and the future cross-session
/// tool that pairs a base and head — goes through this one loader so the read +
/// version-gate + parse policy stays in one place rather than re-derived per call site.
///
/// Distinct from [`detect_schema_version`], which deliberately does *not* deserialize the
/// whole artifact (so it can inspect foreign versions cheaply). Here we want the typed
/// model, so we version-gate first (turning a mismatch into the actionable
/// [`ClsError::SchemaVersionMismatch`] / `CLS-E0003` rather than a noisy serde error) and
/// then parse into [`ClsArtifact`].
pub fn load_artifact(path: &Path) -> Result<ClsArtifact, ClsError> {
    detect_schema_version(path)?;
    let text = std::fs::read_to_string(path).map_err(|source| ClsError::IoError {
        path: path.display().to_string(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| ClsError::SchemaParseError { source })
}
