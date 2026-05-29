//! `cl` — compile-lens analyzer CLI.
//!
//! This is the Rust-side entry point; the Python-side `cl` console script (a separate
//! entry installed by the Python package) fronts the user-facing hero flow and invokes
//! this binary as a subprocess (the design doc ADR-006 *Clarification*: control-flow crosses
//! the language boundary via subprocess + file, not FFI).
//!
//! For v0.5.0 the only wired subcommand is `collect`, kept minimal so the `cls-errors`
//! rendering pipeline (miette + thiserror, ADR-022) is exercised end-to-end. The real
//! collector lands in Phase 1.

use clap::{Parser, Subcommand};
use cls_errors::ClsError;

/// torch.compile production diagnostics.
#[derive(Parser)]
#[command(name = "compile-lens", bin_name = "cl", version = "0.5.0", about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Collect a `.cls.json` session artifact (Phase 1; v0.5.0 is a placeholder
    /// that verifies the input path is readable so the error pipeline is exercised).
    Collect {
        /// Path to read.
        path: std::path::PathBuf,
    },

    /// Migrate an older `.cls.json` to the current schema. Pre-V1 there is no
    /// migration ladder yet: matching the current schema -> byte-copy; otherwise
    /// the migration is refused (CLS-E0003) and the user re-collects.
    Migrate {
        /// Artifact to read.
        input: std::path::PathBuf,
        /// Write the migrated artifact here.
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Report what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> miette::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Collect { path }) => {
            // Skeleton: just open the path. A real read failure round-trips through
            // `ClsError::IoError` (`CLS-E0001`) and miette renders the fancy diagnostic.
            std::fs::read(&path).map_err(|source| ClsError::IoError {
                path: path.display().to_string(),
                source,
            })?;
            println!("collected (placeholder) — {}", path.display());
            Ok(())
        }
        Some(Command::Migrate { input, output, dry_run }) => {
            if dry_run {
                let version = cls_schema_migrate::detect_schema_version(&input)?;
                println!("no changes (artifact is already at schema {version})");
                Ok(())
            } else if let Some(out) = output {
                cls_schema_migrate::migrate_to_current(&input, &out)?;
                println!("migrated {} -> {}", input.display(), out.display());
                Ok(())
            } else {
                Err(ClsError::InvalidCliArgs {
                    detail: "either --output <path> or --dry-run is required".into(),
                }
                .into())
            }
        }
        None => Ok(()),
    }
}
