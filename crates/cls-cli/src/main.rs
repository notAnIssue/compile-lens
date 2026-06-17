//! `cl` — compile-lens analyzer CLI.
//!
//! This is the Rust-side entry point; the Python-side `cl` console script (a separate
//! entry installed by the Python package) fronts the user-facing hero flow and invokes
//! this binary as a subprocess (control-flow crosses the language boundary via
//! subprocess + file, not FFI).
//!
//! Every analyzer subcommand is wired (`recompile-summary` / `diff` / `cache-stability` /
//! `divergence-view` / `compile-lint` / `kernel-roofline` / `fusion-detect`), plus `session
//! report` (the hero HTML), `scrub`, and `migrate`. The one exception is `collect`, which stays a
//! typed `NotYetImplemented` stub: collection is driven from the Python `cl.session()` front-end
//! (the source of truth), and this binary renders/analyzes what it wrote — the data crosses the
//! language boundary as a `.cls.json` file over a subprocess, not via FFI (ADR-006).
//!
//! Observability (tracing + debug only, no metric export) is
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

    /// Analyze a session's candidate lint findings (Tool 4): join them with the
    /// correctness database, escalate severity, and render. Exits 1 if any `high`
    /// finding survives, so it can gate CI. `--format sarif` emits SARIF 2.1.0 for
    /// GitHub Code Scanning. (The Python front-end scans source into the session;
    /// this is the analysis half.)
    CompileLint {
        /// The collected `.cls.json` session whose `lint_findings` to analyze.
        session: std::path::PathBuf,

        /// The correctness-database TOML. Absent → an empty database, so every
        /// candidate (having no cited issue) is dropped.
        #[arg(long, value_name = "PATH")]
        db: Option<std::path::PathBuf>,

        /// Output format: `markdown` (default), `json`, or `sarif`.
        #[arg(long, value_enum, default_value_t = LintFormat::Markdown)]
        format: LintFormat,

        /// Drop findings below this severity. compile-lint is binary: `info` < `high`.
        #[arg(long, value_name = "LEVEL")]
        min_severity: Option<String>,
    },

    /// Kernel roofline triage (Tool 5). For each kernel in a session, compute the theoretical
    /// lower bound (Layer 1) and the empirical predictor (Layer 2) against a GPU spec; when the
    /// kernels carry measured runtimes, calibrate the predicted ranking against the measured one
    /// (Spearman) and report a grid-level pruning decision (Layer 3 — how many configs an
    /// autotuner must actually measure). The autotune harness drives this via `--format json`.
    KernelRoofline {
        /// The collected `.cls.json` session whose `kernels[]` to analyze.
        session: std::path::PathBuf,

        /// GPU spec to compute the roofline against (registry name, case-insensitive).
        #[arg(long, default_value = "A100-SXM-80GB")]
        gpu: String,

        /// Restrict the rendered report to kernels whose id contains this substring.
        #[arg(long, value_name = "SUBSTR")]
        kernel: Option<String>,

        /// List the session's kernel ids and names, then exit without analyzing.
        #[arg(long)]
        list: bool,

        /// Target number of configs to keep under aggressive pruning (Layer 3).
        #[arg(long, default_value_t = 5)]
        top_k: usize,

        /// Output format: `markdown` (default) or `json`.
        #[arg(long, value_enum, default_value_t = SummaryFormat::Markdown)]
        format: SummaryFormat,
    },

    /// Detect algebraic fusion opportunities in a collected session (Tool 6 — the CODA-style
    /// GEMM-Residual-RMSNorm-GEMM detector). Reads the serialized FX graph, matches the pattern,
    /// estimates baseline-vs-fused HBM traffic, and suggests an epilogue kernel — ranked by
    /// estimated speedup. Suggest-only: it will not modify your graph or call any kernel.
    FusionDetect {
        /// The collected `.cls.json` session whose compiled graph(s) to analyze.
        session: std::path::PathBuf,

        /// Output format: `markdown` (default) or `json`.
        #[arg(long, value_enum, default_value_t = SummaryFormat::Markdown)]
        format: SummaryFormat,

        /// Drop opportunities whose estimated (memory-bound) speedup is below this. An opportunity
        /// whose tensor shapes weren't captured has no estimate and is always kept.
        #[arg(long, value_name = "FLOAT", default_value_t = 1.05)]
        min_speedup: f64,

        /// Keep at most this many opportunities, after ranking by estimated speedup.
        #[arg(long, value_name = "N", default_value_t = 10)]
        top_k: usize,
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

    /// Hero report. `cl session report <session.cls.json>` renders a collected session into the
    /// self-contained HTML report (Phase 7, the source of truth is the Python `cl.session()`).
    Session {
        #[command(subcommand)]
        action: SessionCommand,
    },

    /// Sanitize a `.cls.json` artifact before sharing — promote it to a stricter redaction
    /// level (scrub paths / argv tokens / kernel sources / host). Promotion is one-way: a
    /// request to *demote* (a less strict `--to` than the artifact already is) is refused
    /// (CLS-E0009), because redaction is lossy and raw data can only return by re-collecting.
    Scrub {
        /// The `.cls.json` artifact to sanitize (or audit, with `--verify`).
        file: std::path::PathBuf,

        /// Audit mode: report whether the artifact already meets a level's share-safety
        /// guarantees, without modifying it. Exits non-zero if any leak remains. Pairs with
        /// `--target`; ignores the write flags.
        #[arg(long)]
        verify: bool,

        /// Target redaction level. In write mode, defaults to at least `default-strict` (never
        /// demotes). In `--verify` mode, the level to audit against (defaults to the artifact's
        /// own declared policy).
        #[arg(
            long = "to",
            visible_alias = "target",
            value_enum,
            value_name = "LEVEL"
        )]
        to: Option<RedactionLevel>,

        /// Write the sanitized artifact here. Without it the file is sanitized in place.
        #[arg(short, long, value_name = "PATH")]
        output: Option<std::path::PathBuf>,

        /// Report what would change without writing anything.
        #[arg(long)]
        dry_run: bool,

        /// Also scrub long base64/hex argv runs as possible novel secrets (more aggressive,
        /// can over-match a legitimate long argument).
        #[arg(long)]
        strict: bool,
    },
}

/// Subcommands under `cl session`.
#[derive(Subcommand)]
enum SessionCommand {
    /// Render a session's `.cls.json` into the self-contained HTML report.
    Report {
        /// The `.cls.json` artifact to render (the "head" session in diff mode).
        session: std::path::PathBuf,
        /// Optional earlier `.cls.json` to diff against. When given, the report leads with an
        /// IR Diff (base → head) section — the regression view of what the change altered.
        #[arg(long, value_name = "PATH")]
        base: Option<std::path::PathBuf>,
        /// GPU to compute the roofline section against. Defaults to the project baseline.
        #[arg(long, default_value = "A100-SXM-80GB")]
        gpu: String,
        /// Optional correctness database. When given, lint candidates are escalated to
        /// severity-rated findings with issue links; without it the report shows raw candidates.
        #[arg(long, value_name = "PATH")]
        db: Option<std::path::PathBuf>,
        /// Where to write the HTML. Defaults to stdout.
        #[arg(short, long, value_name = "PATH")]
        output: Option<std::path::PathBuf>,
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

impl From<RedactionLevel> for cls_schema::RedactionPolicy {
    fn from(level: RedactionLevel) -> Self {
        match level {
            RedactionLevel::DefaultStrict => cls_schema::RedactionPolicy::DefaultStrict,
            RedactionLevel::Internal => cls_schema::RedactionPolicy::Internal,
            RedactionLevel::Confidential => cls_schema::RedactionPolicy::Confidential,
            RedactionLevel::PublicSafe => cls_schema::RedactionPolicy::PublicSafe,
        }
    }
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

/// Output format for `cl compile-lint --format`. Adds `sarif` (GitHub Code Scanning) on top of the
/// usual markdown/json.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum LintFormat {
    Markdown,
    Json,
    Sarif,
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

        Some(Command::CompileLint {
            session,
            db,
            format,
            min_severity,
        }) => compile_lint(session, db, format, min_severity),

        Some(Command::KernelRoofline {
            session,
            gpu,
            kernel,
            list,
            top_k,
            format,
        }) => kernel_roofline(session, gpu, kernel, list, top_k, format),

        Some(Command::FusionDetect {
            session,
            format,
            min_speedup,
            top_k,
        }) => fusion_detect(session, format, min_speedup, top_k),

        Some(Command::Migrate {
            input,
            output,
            dry_run,
        }) => migrate(input, output, dry_run),

        Some(Command::Session { action }) => match action {
            SessionCommand::Report {
                session,
                base,
                gpu,
                db,
                output,
            } => session_report(session, base, gpu, db, output),
        },

        Some(Command::Scrub {
            file,
            verify,
            to,
            output,
            dry_run,
            strict,
        }) => {
            if verify {
                scrub_verify(file, to)
            } else {
                scrub(file, to, output, dry_run, strict)
            }
        }

        None => Ok(()),
    }
}

/// `cl session report` — render a collected session into the self-contained HTML report (Phase 7).
/// Reads + version-gates the artifact (the shared loader), renders via `cls-report`, and writes the
/// HTML to `--output` or stdout. Pure Rust: the Python `cl.session()` controls collection; this
/// renders what it wrote, the data crossing as the `.cls.json` file (ADR-006), never FFI.
/// Pruning target for the report's roofline section. The section is informational, not an autotune
/// gate, so this figure only shapes the Layer-3 pruning line when measurements are present.
const REPORT_ROOFLINE_TOP_K: usize = 8;

fn session_report(
    session: std::path::PathBuf,
    base: Option<std::path::PathBuf>,
    gpu: String,
    db: Option<std::path::PathBuf>,
    output: Option<std::path::PathBuf>,
) -> Result<(), ClsError> {
    let artifact = cls_schema_migrate::load_artifact(&session)?;
    // In `--base` mode, compute the base→head compile diff (Tool 2a) and render it as the pillar
    // IR Diff section. We diff the first compiled graph on each side, matching `cl diff`; an
    // artifact with no compiled graph diffs as an empty graph rather than erroring.
    let diff = match &base {
        Some(base_path) => {
            let base_artifact = cls_schema_migrate::load_artifact(base_path)?;
            let before = cls_wl_diff::FxGraph::from_nodes(first_graph_nodes(&base_artifact));
            let after = cls_wl_diff::FxGraph::from_nodes(first_graph_nodes(&artifact));
            Some(cls_wl_diff::diff_graphs(&before, &after))
        }
        None => None,
    };
    // Lint escalates only with a `--db`; without one the renderer shows the raw candidates and how
    // to escalate (an empty DB would silently drop every candidate, so we don't analyze at all).
    let lint = match db.as_deref() {
        Some(path) => {
            let pattern_db = load_pattern_db(Some(path))?;
            Some(cls_analyzer::lint::analyze(&artifact, &pattern_db))
        }
        None => None,
    };
    // Roofline is always computed against `--gpu` (default = project baseline), so the section shows
    // real numbers by default; an unknown GPU name errors here rather than silently dropping it.
    let roofline = cls_analyzer::roofline::analyze(&artifact, &gpu, REPORT_ROOFLINE_TOP_K)?;
    // Fusion (Tool 6) is derived from the captured graphs, not stored on the artifact, so the report
    // runs the analyzer the same way it runs lint/roofline — otherwise the crown-jewel section would
    // always render empty for a freshly captured session.
    let fusion = cls_analyzer::fusion::analyze(&artifact);
    let inputs = cls_report::ReportInputs {
        diff: diff.as_ref(),
        lint: lint.as_ref(),
        roofline: Some(&roofline),
        fusion: Some(&fusion),
    };
    let html = cls_report::render(&artifact, &inputs);
    match output {
        Some(path) => {
            std::fs::write(&path, html).map_err(|source| ClsError::IoError {
                path: path.display().to_string(),
                source,
            })?;
            println!("wrote {}", path.display());
        }
        None => print!("{html}"),
    }
    Ok(())
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

/// `cl cache-stability` (Tool 2b, Mode B). Load a session and flag the silently-wrong cache bug:
/// iterations where the module's mutable state drifted, the compiled graph was reused from cache,
/// and the output stayed frozen — the cache key missed the state change and served a stale graph.
/// Reads the per-iteration data; renders markdown|json (Text falls back to Markdown).
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

/// `cl kernel-roofline` — Tool 5. Load a session, compute each kernel's roofline prediction
/// (Layer 1 + Layer 2) against `--gpu`, and — when measurements are present — a grid-level
/// calibration plus pruning decision (Layer 3). `--list` short-circuits to a kernel listing;
/// `--kernel` narrows the rendered report to matching ids. The JSON shape is what the autotune
/// harness consumes across the subprocess boundary (ADR-006).
fn kernel_roofline(
    session: std::path::PathBuf,
    gpu: String,
    kernel: Option<String>,
    list: bool,
    top_k: usize,
    format: SummaryFormat,
) -> Result<(), ClsError> {
    let artifact = cls_schema_migrate::load_artifact(&session)?;

    if list {
        for k in &artifact.kernels {
            println!("{}\t{}", k.kernel_id, k.name);
        }
        return Ok(());
    }

    let mut report = cls_analyzer::roofline::analyze(&artifact, &gpu, top_k)?;
    if let Some(substr) = &kernel {
        report.predictions.retain(|p| p.kernel_id.contains(substr));
    }
    let render_format = match format {
        SummaryFormat::Json => cls_analyzer::roofline::Format::Json,
        SummaryFormat::Markdown | SummaryFormat::Text => cls_analyzer::roofline::Format::Markdown,
    };
    print!("{}", cls_analyzer::roofline::render(&report, render_format));
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

/// `cl compile-lint`: join the session's candidate `lint_findings` with the correctness database,
/// render in the chosen format, and exit 1 if any `high` finding survives so the command can gate
/// CI. The Python front-end runs the source scan that fills `lint_findings`; this is the
/// analysis-and-report half — the two meet on disk (ADR-006), never over FFI.
fn compile_lint(
    session: std::path::PathBuf,
    db: Option<std::path::PathBuf>,
    format: LintFormat,
    min_severity: Option<String>,
) -> Result<(), ClsError> {
    let artifact = cls_schema_migrate::load_artifact(&session)?;
    let pattern_db = load_pattern_db(db.as_deref())?;
    let mut report = cls_analyzer::lint::analyze(&artifact, &pattern_db);

    if let Some(floor) = min_severity.as_deref() {
        let floor_rank = severity_rank(floor).ok_or_else(|| ClsError::InvalidCliArgs {
            detail: format!("--min-severity `{floor}` must be one of: info, high"),
        })?;
        report
            .findings
            .retain(|f| severity_rank(&f.severity).unwrap_or(0) >= floor_rank);
    }

    let rendered = match format {
        LintFormat::Sarif => {
            cls_sarif::to_sarif_json(&report.findings, "compile-lens", env!("CARGO_PKG_VERSION"))
        }
        LintFormat::Json => cls_analyzer::lint::render(&report, cls_analyzer::lint::Format::Json),
        LintFormat::Markdown => {
            cls_analyzer::lint::render(&report, cls_analyzer::lint::Format::Markdown)
        }
    };
    print!("{rendered}");

    // CI gate: any surviving `high` finding fails the run with exit 1 — kept distinct from the
    // typed-error exit code 3 so a script can tell "lint found a real bug" from "lint itself errored".
    if report.findings.iter().any(|f| f.severity == "high") {
        process::exit(1);
    }
    Ok(())
}

/// Load the correctness database from a TOML path, or an empty database when `--db` is absent. An
/// empty database drops every candidate (no cited issue → not a finding), which is the
/// lint-not-oracle default: with no evidence to stand on, the tool reports nothing.
fn load_pattern_db(
    path: Option<&std::path::Path>,
) -> Result<cls_correctness_db::PatternDb, ClsError> {
    let Some(path) = path else {
        return Ok(cls_correctness_db::PatternDb::default());
    };
    let toml = std::fs::read_to_string(path).map_err(|source| ClsError::IoError {
        path: path.display().to_string(),
        source,
    })?;
    cls_correctness_db::PatternDb::from_toml(&toml).map_err(|source| ClsError::InvalidCliArgs {
        detail: format!(
            "--db `{}` is not a valid correctness database: {source}",
            path.display()
        ),
    })
}

/// Rank a severity label for `--min-severity` comparison; `None` for an unrecognized label so a
/// typo'd floor is rejected (CLS-E0004) rather than silently passing every finding. compile-lint
/// emits only `info` and `high` (a finding is `high` unless its issue is already fixed for the
/// user's torch version, then `info`), so those are the only two floors — there is no `warning`.
fn severity_rank(severity: &str) -> Option<u8> {
    match severity {
        "high" => Some(2),
        "info" => Some(1),
        _ => None,
    }
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

/// `cl fusion-detect` — Tool 6. Load a session, match CODA-style fusion opportunities in its
/// compiled graph(s), estimate baseline-vs-fused HBM traffic, and render the report ranked by
/// estimated speedup (`--min-speedup` filters, `--top-k` caps). Suggest-only and read-only: it never
/// mutates the graph, runs a kernel, or needs torch — it reads the serialized FX graph the collector
/// already wrote (ADR-037). Always exits 0 on a successful read: it is informational, not a CI gate.
fn fusion_detect(
    session: std::path::PathBuf,
    format: SummaryFormat,
    min_speedup: f64,
    top_k: usize,
) -> Result<(), ClsError> {
    let artifact = cls_schema_migrate::load_artifact(&session)?;
    let opportunities = cls_analyzer::fusion::analyze(&artifact);
    let selected = cls_analyzer::fusion::select(opportunities, min_speedup, top_k);
    // markdown|json; Text falls back to Markdown, as the other view subcommands do.
    let render_format = match format {
        SummaryFormat::Json => cls_analyzer::fusion::Format::Json,
        SummaryFormat::Markdown | SummaryFormat::Text => cls_analyzer::fusion::Format::Markdown,
    };
    print!("{}", cls_analyzer::fusion::render(&selected, render_format));
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

/// `cl scrub <file>` — promote a `.cls.json` artifact to a stricter redaction level before
/// sharing. Loads + version-gates the artifact (the shared loader), runs the `cls-scrub` rules,
/// and writes the sanitized JSON (in place, or to `--output`). `--dry-run` reports the change
/// without writing. Demotion is refused inside `redact_artifact` (CLS-E0009).
fn scrub(
    file: std::path::PathBuf,
    to: Option<RedactionLevel>,
    output: Option<std::path::PathBuf>,
    dry_run: bool,
    strict: bool,
) -> Result<(), ClsError> {
    // An HTML report takes a different path: it ensures the strict CSP, re-scrubs leaked tokens,
    // and neutralizes dangerous constructs — not the JSON field rules. (Redaction levels don't
    // apply to rendered HTML, so `--to` / `--strict` are ignored here.)
    if file
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("html"))
    {
        return scrub_html_file(file, output, dry_run);
    }

    let mut artifact = cls_schema_migrate::load_artifact(&file)?;
    let current = artifact.session.redaction_policy;
    // Default target is "at least default-strict, never demote" so `cl scrub <file>` with no
    // flag always lands on a share-safe artifact and an already-stricter one keeps its level.
    let target = match to {
        Some(level) => level.into(),
        None => cls_scrub::stricter_of(current, cls_schema::RedactionPolicy::DefaultStrict),
    };

    let salt = cls_scrub::rules::install_salt().map_err(|source| ClsError::IoError {
        path: "~/.compile-lens/install-id".into(),
        source,
    })?;

    let stats = cls_scrub::redact_artifact(&mut artifact, target, &salt, strict)?;
    let summary = format_scrub_summary(&stats);
    let level = cls_scrub::level_name(target);

    if dry_run {
        println!("would scrub {} -> {level} ({summary})", file.display());
        return Ok(());
    }

    let json =
        serde_json::to_string(&artifact).map_err(|source| ClsError::SchemaParseError { source })?;
    let dest = output.as_deref().unwrap_or(&file);
    std::fs::write(dest, json).map_err(|source| ClsError::IoError {
        path: dest.display().to_string(),
        source,
    })?;
    println!("✅ wrote {} ({level}: {summary})", dest.display());
    Ok(())
}

/// `cl scrub <report.html>` — sanitize an HTML report for sharing: ensure the strict CSP, re-scrub
/// leaked secret tokens, neutralize any literal `<script>` / `on*=` / `javascript:`. Writes in
/// place or to `--output`; `--dry-run` reports without writing.
fn scrub_html_file(
    file: std::path::PathBuf,
    output: Option<std::path::PathBuf>,
    dry_run: bool,
) -> Result<(), ClsError> {
    let html = std::fs::read_to_string(&file).map_err(|source| ClsError::IoError {
        path: file.display().to_string(),
        source,
    })?;
    let (sanitized, stats) = cls_scrub::html::scrub_html(&html);
    let summary = format_html_summary(&stats);

    if dry_run {
        println!("would scrub {} ({summary})", file.display());
        return Ok(());
    }

    let dest = output.as_deref().unwrap_or(&file);
    std::fs::write(dest, sanitized).map_err(|source| ClsError::IoError {
        path: dest.display().to_string(),
        source,
    })?;
    println!("✅ wrote {} ({summary})", dest.display());
    Ok(())
}

/// One-line summary of an HTML scrub, listing only what changed.
fn format_html_summary(stats: &cls_scrub::html::HtmlScrubStats) -> String {
    if stats.is_noop() {
        return "already share-safe".to_string();
    }
    let mut parts = Vec::new();
    if stats.csp_added {
        parts.push("CSP added".to_string());
    }
    if stats.tokens_scrubbed {
        parts.push("tokens scrubbed".to_string());
    }
    if stats.scripts_neutralized > 0 {
        parts.push(format!(
            "{} script tag(s) neutralized",
            stats.scripts_neutralized
        ));
    }
    if stats.handlers_neutralized > 0 {
        parts.push(format!(
            "{} event handler(s) removed",
            stats.handlers_neutralized
        ));
    }
    if stats.js_uris_neutralized > 0 {
        parts.push(format!(
            "{} javascript: URI(s) blocked",
            stats.js_uris_neutralized
        ));
    }
    parts.join(", ")
}

/// `cl scrub --verify <file> [--target LEVEL]` — audit whether the artifact already meets a
/// level's share-safety guarantees, without modifying it. Prints a ✅/⚠️ line per category and a
/// verdict, then exits 1 if any leak remains (so CI / a release gate can block on it). The target
/// defaults to the artifact's own declared policy.
fn scrub_verify(file: std::path::PathBuf, target: Option<RedactionLevel>) -> Result<(), ClsError> {
    let artifact = cls_schema_migrate::load_artifact(&file)?;
    let target = target
        .map(Into::into)
        .unwrap_or(artifact.session.redaction_policy);
    let report = cls_scrub::verify(&artifact, target);

    println!(
        "artifact policy: {} (target: {})",
        cls_scrub::level_name(report.declared),
        cls_scrub::level_name(report.target),
    );
    for check in &report.checks {
        println!(
            "{} {}",
            if check.passed { "✅" } else { "⚠️" },
            check.message
        );
    }

    if report.is_clean() {
        println!(
            "→ SHARE-SAFE for {} distribution",
            cls_scrub::level_name(report.target)
        );
        Ok(())
    } else {
        println!(
            "→ NOT {}; run: cl scrub {} --to {}",
            cls_scrub::level_name(report.target),
            file.display(),
            cls_scrub::level_name(report.target),
        );
        // Distinct from the typed-error exit codes (3..): a successful audit that *found* a leak
        // exits 1, like `cl compile-lint` exiting 1 on a surviving high finding, so a script can
        // tell "verify ran and flagged a leak" from "verify itself errored".
        process::exit(1);
    }
}

/// Build the operator-facing one-line summary of a scrub, listing only the parts that changed
/// (e.g. `host redacted, 1 argv token scrubbed, 2 kernel sources removed`). "no changes" when
/// the artifact was already at or above the target's guarantees.
fn format_scrub_summary(stats: &cls_scrub::RedactionStats) -> String {
    fn plural(n: usize, noun: &str) -> String {
        format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
    }
    let mut parts = Vec::new();
    if stats.host_redacted {
        parts.push("host redacted".to_string());
    }
    if stats.argv_tokens_scrubbed {
        parts.push("argv scrubbed".to_string());
    }
    if stats.paths_normalized > 0 {
        parts.push(format!(
            "{} normalized",
            plural(stats.paths_normalized, "path")
        ));
    }
    if stats.kernel_sources_removed > 0 {
        parts.push(format!(
            "{} removed",
            plural(stats.kernel_sources_removed, "kernel source")
        ));
    }
    if stats.source_excerpts_removed > 0 {
        parts.push(format!(
            "{} removed",
            plural(stats.source_excerpts_removed, "source excerpt")
        ));
    }
    if stats.env_vars_dropped > 0 {
        parts.push(format!(
            "{} dropped",
            plural(stats.env_vars_dropped, "env var")
        ));
    }
    if parts.is_empty() {
        "no changes".to_string()
    } else {
        parts.join(", ")
    }
}
