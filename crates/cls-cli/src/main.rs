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
//!
//! Observability (the design doc / D9 — tracing + debug only, no metric export) is
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

    /// Diff two collected sessions' compiled graphs (Tool 2a — WL-signature
    /// neighborhood diff). Reports the nodes a change added / removed / modified
    /// between a `--base` and a `--head` artifact, with a per-match confidence and
    /// two quality numbers (match coverage and anchor uniqueness), rendered as
    /// `markdown`, `json`, or `text`.
    Diff {
        /// The baseline `.cls.json` artifact (the "before" side).
        #[arg(long, value_name = "PATH")]
        base: std::path::PathBuf,

        /// The head `.cls.json` artifact (the "after" side).
        #[arg(long, value_name = "PATH")]
        head: std::path::PathBuf,

        /// Output format. `markdown` is the default human-readable shape;
        /// `json` is machine-readable; `text` is plain console output for
        /// terminals that can't render Markdown.
        #[arg(long, value_enum, default_value_t = SummaryFormat::Markdown)]
        format: SummaryFormat,
    },

    /// Detect cache-stability anomalies in a single collected session (Tool 2b,
    /// Mode B). Flags iterations where the module's internal state drifted, the
    /// compiled graph was reused from cache, and the output stayed frozen — a
    /// silently-wrong mutable-state-not-invalidated bug (Li et al. 2026, Listing 2).
    CacheStability {
        /// The collected `.cls.json` session to analyze.
        session: std::path::PathBuf,

        /// Output format: `markdown` (default) or `json`.
        #[arg(long, value_enum, default_value_t = SummaryFormat::Markdown)]
        format: SummaryFormat,
    },

    /// View Tool 3's eager-vs-compiled divergence findings already stored in a
    /// `.cls.json` (ADR-034). View-only: it reads and renders the captured
    /// `divergences[]` and never re-runs the model, so it needs no torch.
    DivergenceView {
        /// The collected `.cls.json` session to view.
        session: std::path::PathBuf,

        /// Output format: `markdown` (default) or `json`.
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

        Some(Command::Diff { base, head, format }) => diff(base, head, format),

        Some(Command::CacheStability { session, format }) => cache_stability(session, format),

        Some(Command::DivergenceView { session, format }) => divergence_view(session, format),

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

/// `cl diff` skeleton (Tool 2a). Validates that both the `--base` and `--head`
/// artifacts are reachable, then exits with the typed `NotYetImplemented` error.
/// The WL-signature diff algorithm (`cls-wl-diff`) lands in upcoming Phase 2 PRs.
fn cache_stability(session: std::path::PathBuf, format: SummaryFormat) -> Result<(), ClsError> {
    // `load_artifact` reads + version-gates + parses; a missing file surfaces as CLS-E0001.
    let artifact = cls_schema_migrate::load_artifact(&session)?;
    let findings = cls_analyzer::cache_stability::analyze(&artifact);
    // The subcommand exposes markdown|json; Text falls back to Markdown.
    let render_format = match format {
        SummaryFormat::Json => cls_analyzer::cache_stability::Format::Json,
        SummaryFormat::Markdown | SummaryFormat::Text => {
            cls_analyzer::cache_stability::Format::Markdown
        }
    };
    print!(
        "{}",
        cls_analyzer::cache_stability::render(&findings, render_format)
    );
    Ok(())
}

/// `cl divergence-view`: load a `.cls.json` and render its stored `divergences[]` (ADR-034).
///
/// View-only — there is no analysis step and no torch: the eager-vs-compiled comparison ran when
/// the artifact was written, so viewing a prior session is a pure read-and-format of the records.
fn divergence_view(session: std::path::PathBuf, format: SummaryFormat) -> Result<(), ClsError> {
    // `load_artifact` reads + version-gates + parses; a missing file surfaces as CLS-E0001.
    let artifact = cls_schema_migrate::load_artifact(&session)?;
    let render_format = match format {
        SummaryFormat::Json => cls_analyzer::divergence::Format::Json,
        SummaryFormat::Markdown | SummaryFormat::Text => cls_analyzer::divergence::Format::Markdown,
    };
    print!(
        "{}",
        cls_analyzer::divergence::render(&artifact.divergences, render_format)
    );
    Ok(())
}

fn diff(
    base: std::path::PathBuf,
    head: std::path::PathBuf,
    format: SummaryFormat,
) -> Result<(), ClsError> {
    // `load_artifact` reads + version-gates + parses each side; a missing file surfaces as the
    // typed `IoError` (CLS-E0001, exit 3) here, not deep in the diff.
    let before = cls_schema_migrate::load_artifact(&base)?;
    let after = cls_schema_migrate::load_artifact(&head)?;

    // Diff the first compiled graph on each side (a session captures one graph per compile). An
    // artifact with no compiled graph diffs as an empty graph rather than erroring.
    let before_graph = cls_wl_diff::FxGraph::from_nodes(first_graph_nodes(&before));
    let after_graph = cls_wl_diff::FxGraph::from_nodes(first_graph_nodes(&after));
    let graph_diff = cls_wl_diff::diff_graphs(&before_graph, &after_graph);

    // `cl diff` also carries the cache-stability diff (Tool 2b Mode A): a regression in cache
    // behavior the change introduced. On graph-only sessions (no `iterations[]`) this is clean.
    let cache_stability = cls_analyzer::cache_stability::analyze_diff(&before, &after);

    match format {
        SummaryFormat::Json => {
            // Combined machine-readable contract: the graph diff and the cache-stability diff.
            let report = DiffReport {
                graph_diff: &graph_diff,
                cache_stability: &cache_stability,
            };
            print!(
                "{}",
                serde_json::to_string_pretty(&report).expect("DiffReport serializes")
            );
        }
        SummaryFormat::Markdown | SummaryFormat::Text => {
            let graph_format = match format {
                SummaryFormat::Text => cls_wl_diff::Format::Text,
                _ => cls_wl_diff::Format::Markdown,
            };
            print!("{}", cls_wl_diff::render(&graph_diff, graph_format));
            print!(
                "\n{}",
                cls_analyzer::cache_stability::render_diff_markdown(&cache_stability)
            );
        }
    }
    Ok(())
}

/// The combined `cl diff --format json` payload: the graph diff plus the cache-stability diff.
#[derive(serde::Serialize)]
struct DiffReport<'a> {
    graph_diff: &'a cls_wl_diff::IrGraphDiff,
    cache_stability: &'a cls_analyzer::cache_stability::CacheStabilityDiff,
}

/// The node-level structure of an artifact's first compiled graph, or empty if it has none.
fn first_graph_nodes(artifact: &cls_schema::ClsArtifact) -> &[cls_schema::FxNode] {
    artifact
        .compiled_graphs
        .first()
        .map(|g| g.nodes.as_slice())
        .unwrap_or(&[])
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
