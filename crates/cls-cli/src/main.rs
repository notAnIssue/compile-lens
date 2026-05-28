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
        None => Ok(()),
    }
}
