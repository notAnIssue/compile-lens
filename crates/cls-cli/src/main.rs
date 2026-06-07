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

use clap::{Parser, Subcommand, ValueEnum};
use cls_errors::ClsError;
use std::process;
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
    /// Collect a `.cls.json` session artifact. The mode flags
    /// (`--from-logs` / `--from-tlparse` / `--from-dynamo-explain`) are
    /// mutually exclusive and one is required. The collectors themselves
    /// land in the upcoming Tool 1 PRs; right now the subcommand parses
    /// arguments, verifies the input path is reachable when applicable,
    /// and exits with the typed `NotYetImplemented` error.
    Collect {
        /// Mode A — parse a `TORCH_LOGS=+recompiles` capture from a file.
        #[arg(long, value_name = "PATH", group = "mode")]
        from_logs: Option<std::path::PathBuf>,

        /// Mode B — read `tlparse` output from a directory.
        #[arg(long, value_name = "DIR", group = "mode")]
        from_tlparse: Option<std::path::PathBuf>,

        /// Mode C — invoke `torch._dynamo.explain` against the workload.
        /// (Programmatic; no path argument.)
        #[arg(long, group = "mode")]
        from_dynamo_explain: bool,

        /// Where to write the collected `.cls.json` session artifact.
        #[arg(short, long, value_name = "PATH")]
        output: std::path::PathBuf,

        /// Iteration count placeholder for Phase 2 (sliding-window collection).
        /// Ignored in Phase 1 but accepted so scripts written against the v0.6
        /// surface don't fail arg parse.
        #[arg(long, value_name = "N", default_value_t = 1)]
        iterations: u64,

        /// Redaction level applied at capture time.
        #[arg(long, value_enum, default_value_t = RedactionLevel::DefaultStrict)]
        redaction: RedactionLevel,
    },

    /// Render the Tool 1 recompile aggregator's analysis of a collected
    /// session into a human- or machine-readable report. With `--baseline`
    /// it instead diffs the session against an earlier one and reports the
    /// recompiles this change introduced / worsened / fixed (regression mode).
    RecompileSummary {
        /// The `.cls.json` artifact to analyze (the "head" session in diff mode).
        session: std::path::PathBuf,

        /// Optional earlier `.cls.json` to diff against. When given, the output
        /// is a regression diff (added / grown / removed recompile clusters)
        /// rather than a single-session summary.
        #[arg(long, value_name = "PATH")]
        baseline: Option<std::path::PathBuf>,

        /// Output format. `markdown` is the default human-readable shape;
        /// `json` is machine-readable; `text` is plain console output for
        /// terminals that can't render Markdown.
        #[arg(long, value_enum, default_value_t = SummaryFormat::Markdown)]
        format: SummaryFormat,
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

/// Redaction-policy level for `cl collect --redaction`. Mirrors the schema's
/// `RedactionPolicy` enum (`default-strict` / `internal` / `confidential` /
/// `public-safe`) so CLI input round-trips to the artifact unchanged.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum RedactionLevel {
    #[value(name = "default-strict")]
    DefaultStrict,
    #[value(name = "internal")]
    Internal,
    #[value(name = "confidential")]
    Confidential,
    #[value(name = "public-safe")]
    PublicSafe,
}

/// Output format for `cl recompile-summary --format`. Maps onto the analyzer's
/// presentation-agnostic [`cls_analyzer::recompile_render::Format`]; kept as a separate
/// clap `ValueEnum` so the analyzer crate stays free of any CLI dependency.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum SummaryFormat {
    Markdown,
    Json,
    Text,
}

impl From<SummaryFormat> for cls_analyzer::recompile_render::Format {
    fn from(f: SummaryFormat) -> Self {
        match f {
            SummaryFormat::Markdown => Self::Markdown,
            SummaryFormat::Json => Self::Json,
            SummaryFormat::Text => Self::Text,
        }
    }
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

fn main() {
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
    match run(cli) {
        Ok(()) => process::exit(0),
        Err(err) => {
            let code = err.exit_code();
            // Render through miette so the user gets the 4-part diagnostic (code +
            // message + cause chain + help) before the process exits with the
            // variant-specific code.
            eprintln!("{:?}", miette::Report::new(err));
            process::exit(code);
        }
    }
}

/// Dispatch the parsed CLI to the underlying functionality. Returns a
/// `ClsError` (rather than `miette::Report`) so [`main`] can read the typed
/// variant and map it to the per-variant exit code.
fn run(cli: Cli) -> Result<(), ClsError> {
    match cli.command {
        Some(Command::Collect {
            from_logs,
            from_tlparse,
            from_dynamo_explain,
            output: _output,
            iterations: _iterations,
            redaction: _redaction,
        }) => collect(from_logs, from_tlparse, from_dynamo_explain),

        Some(Command::RecompileSummary {
            session,
            baseline,
            format,
        }) => recompile_summary(session, baseline, format),

        Some(Command::Migrate {
            input,
            output,
            dry_run,
        }) => migrate(input, output, dry_run),

        None => Ok(()),
    }
}

/// `cl collect` skeleton. Validates the mode and the input path; the
/// actual collectors land in upcoming Phase 1 PRs.
fn collect(
    from_logs: Option<std::path::PathBuf>,
    from_tlparse: Option<std::path::PathBuf>,
    from_dynamo_explain: bool,
) -> Result<(), ClsError> {
    // clap's `group = "mode"` makes the three options mutually exclusive; this
    // branch tells the user one of them is required if none were given.
    let surface = match (&from_logs, &from_tlparse, from_dynamo_explain) {
        (Some(path), None, false) => {
            // Mode-A inputs are file paths; verify the file is readable now so
            // a typo surfaces as the typed `IoError` (CLS-E0001) with exit code
            // 3 instead of getting swallowed by the not-yet-implemented branch.
            std::fs::metadata(path).map_err(|source| ClsError::IoError {
                path: path.display().to_string(),
                source,
            })?;
            "cl collect --from-logs"
        }
        (None, Some(dir), false) => {
            std::fs::metadata(dir).map_err(|source| ClsError::IoError {
                path: dir.display().to_string(),
                source,
            })?;
            "cl collect --from-tlparse"
        }
        (None, None, true) => "cl collect --from-dynamo-explain",
        (None, None, false) => {
            return Err(ClsError::InvalidCliArgs {
                detail: "one of --from-logs / --from-tlparse / --from-dynamo-explain is required"
                    .into(),
            });
        }
        // clap's `group` should make these unreachable, but the typed error is
        // still the right answer if the surface ever drifts.
        _ => {
            return Err(ClsError::InvalidCliArgs {
                detail: "--from-logs, --from-tlparse, --from-dynamo-explain are mutually exclusive"
                    .into(),
            });
        }
    };

    Err(ClsError::NotYetImplemented {
        surface: surface.into(),
        tracking: "Phase 1 (Tool 1)".into(),
    })
}

/// `cl recompile-summary` — load a session artifact, run Tool 1, and print the report.
///
/// Two modes share one loader (`cls_schema_migrate::load_artifact`, the ADR-033 §5
/// baseline-pairing seam, which reads + version-gates + deserializes):
/// - no `--baseline`: [`cls_analyzer::recompile::analyze`] → single-session summary.
/// - `--baseline B`: load `B` too and [`cls_analyzer::recompile_diff::diff_recompiles`]
///   `(base, head)` → regression diff.
///
/// The chosen `--format` is mapped onto the analyzer's render `Format` and the rendered
/// string is written to stdout (machine-readable `json` and the human shapes alike go to
/// stdout; only diagnostics use stderr).
fn recompile_summary(
    session: std::path::PathBuf,
    baseline: Option<std::path::PathBuf>,
    format: SummaryFormat,
) -> Result<(), ClsError> {
    let head = cls_schema_migrate::load_artifact(&session)?;
    let render_format = format.into();

    let rendered = match baseline {
        Some(base_path) => {
            let base = cls_schema_migrate::load_artifact(&base_path)?;
            let diff = cls_analyzer::recompile_diff::diff_recompiles(&base, &head);
            cls_analyzer::recompile_render::render_diff(&diff, render_format)
        }
        None => {
            let findings = cls_analyzer::recompile::analyze(&head)?;
            cls_analyzer::recompile_render::render_findings(&findings, render_format)
        }
    };

    print!("{rendered}");
    Ok(())
}

/// `cl migrate` — unchanged from v0.5.0; kept under `run` so the dispatch
/// returns `ClsError` uniformly.
fn migrate(
    input: std::path::PathBuf,
    output: Option<std::path::PathBuf>,
    dry_run: bool,
) -> Result<(), ClsError> {
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
        })
    }
}
