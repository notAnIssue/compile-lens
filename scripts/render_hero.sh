#!/usr/bin/env bash
#
# Render the showcase hero report from the committed example artifacts — one command, no manual
# build, no "which cl" guesswork.
#
# It builds the release `cl` binary (so the report always reflects the current renderer — never a
# stale binary), renders the base→head report from examples/, and confirms it is share-safe. The
# output, examples/hero.html, is committed and CI-checked, so you normally do not need to run this
# at all — just open examples/hero.html. Run this only to regenerate it after a renderer change.
#
# Usage:  ./scripts/render_hero.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CL="$ROOT/target/release/cl"

echo "→ building the release cl binary (current renderer, not a stale one)"
cargo build --release -p cls-cli

echo "→ rendering examples/hero.html from the committed base→head artifacts"
"$CL" session report examples/hero_head.cls.json \
    --base examples/hero_base.cls.json \
    --output examples/hero.html

echo "→ confirming the report is share-safe"
"$CL" scrub examples/hero.html

echo "✓ wrote examples/hero.html — open it in a browser."
