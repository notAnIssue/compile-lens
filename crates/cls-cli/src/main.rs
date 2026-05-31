//! `cl` — compile-lens analyzer CLI.
//!
//! This is the Rust-side entry point; the Python-side `cl` console script (a separate
//! entry installed by the Python package) fronts the user-facing hero flow and invokes
//! this binary as a subprocess (design.md ADR-006 *Clarification*: control-flow crosses
//! the language boundary via subprocess + file, not FFI).
//!
//! For v0.5.0 the only wired subcommand is `collect`, kept minimal so the `cls-errors`
//! rendering pipeline (miette + thiserror, ADR-022) is exercised end-to-end. The real
//! collector lands in Phase 1.
//!
//! Observability (design.md §15.3 / D9 — tracing + debug only, no metric export) is
//! configured by environment variables, *not* by CLI flags, so the surface stays the same
//! for every subprocess invocation made by the Python front-end:
//!
//! - `CLS_LOG` (e.g. `debug` / `info` / `warn`) — verbosity, parsed as a
//!   `tracing_subscriber::EnvFilter` directive. Default: `warn` (so `cl --version` is
//!   silent on stderr in normal use).
//! - `CLS_LOG_FORMAT=json` — emit one JSON object per event (opt-in). Anything else
//!   (incl. unset) gives the default human-readable formatter.

use clap::{Parser, Subcommand};
use cls_errors::ClsError;
use tracing_subscriber::EnvFilter;

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

/// Install a global tracing subscriber from `CLS_LOG` / `CLS_LOG_FORMAT`.
///
/// Called once at the top of `main`, *before* `Cli::parse()` — clap exits the process on
/// `--version` / `--help` without returning, so any startup event has to be emitted before
/// the parser runs or it never surfaces.
fn init_tracing() {
    let filter = EnvFilter::try_from_env("CLS_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
    let json = std::env::var("CLS_LOG_FORMAT")
        .map(|s| s.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    // `.json()` returns a different builder type from the default formatter, so the
    // two branches each consume their builder via `.init()` rather than sharing a binding.
    if json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

fn main() -> miette::Result<()> {
    init_tracing();
    // Two startup events at different levels so the env-var contract is observable from
    // outside without any subcommand running: `CLS_LOG=debug cl --version` surfaces the
    // debug event; `CLS_LOG=info cl --version` (any format) surfaces the info event.
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "cl ready");
    tracing::debug!(
        args = ?std::env::args().collect::<Vec<_>>(),
        "cl invocation"
    );

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
        Some(Command::Migrate {
            input,
            output,
            dry_run,
        }) => {
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
