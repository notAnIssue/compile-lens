//! `cl` — compile-lens command-line entry point (skeleton).

use clap::Parser;

/// torch.compile production diagnostics.
#[derive(Parser)]
#[command(name = "compile-lens", version = "0.5.0", about)]
struct Cli {}

fn main() {
    let _ = Cli::parse();
}
