#!/usr/bin/env bash
#
# Run every check CI runs, locally, before you push.
#
# CI is a gate, not a debugger. If this script is green, the PR should land
# green; if a CI job fails afterward, the discrepancy is itself a bug to fix
# in this script (or in the workflow).
#
# Usage:
#   ./scripts/preflight.sh           # full sweep
#   ./scripts/preflight.sh --quick   # skip integration tests (still runs unit + lint)

set -euo pipefail

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
fi

step() { printf '\n\033[1;34m→\033[0m %s\n' "$*"; }

# Resolve the python tools from .venv if present, else system PATH.
PY_BIN="python3"
PIP_DIR=""
if [[ -d .venv ]]; then
  PIP_DIR=".venv/bin"
  PY_BIN=".venv/bin/python"
fi

step "cargo fmt --check"
cargo fmt --all -- --check

step "cargo clippy -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

step "cargo test --workspace"
cargo test --workspace

step "ruff check"
"${PIP_DIR:+$PIP_DIR/}ruff" check python/

step "ruff format --check"
"${PIP_DIR:+$PIP_DIR/}ruff" format --check python/

step "mypy"
"${PIP_DIR:+$PIP_DIR/}mypy" python/compile_lens

step "pytest python/tests"
PYTHONPATH=python "${PIP_DIR:+$PIP_DIR/}pytest" python/tests -q

if (( QUICK == 0 )); then
  step "pytest tests/integration"
  "${PIP_DIR:+$PIP_DIR/}pytest" tests/integration -q
fi

step "pre-commit run --all-files"
"${PIP_DIR:+$PIP_DIR/}pre-commit" run --all-files

step "private-fingerprint sweep (full tree)"
# This duplicates what the pre-commit hook does on staged files, but across
# the whole tree — catches anything that slipped in earlier.
LEAKS=$(git grep -nE 'colibri/|assets/notes/' -- '*.md' '*.rs' '*.py' '*.toml' '*.yml' \
  2>/dev/null | grep -v -E '(scripts/check_private_fingerprints\.sh|CONTRIBUTING\.md|docs/02_design_decisions/adr-028)' \
  || true)
if [[ -n "$LEAKS" ]]; then
  echo "FAIL: internal planning paths present in tracked files:"
  echo "$LEAKS" | sed 's/^/    /'
  exit 1
fi

SECTION_LEAKS=$(git grep -nE '(^|[^A-Za-z0-9])S[0-9]+\.[0-9XN]+([^A-Za-z0-9]|$)' \
  -- '*.md' '*.rs' '*.py' '*.toml' '*.yml' 2>/dev/null \
  | grep -v -E '(scripts/check_private_fingerprints\.sh|CONTRIBUTING\.md|docs/02_design_decisions/adr-028)' \
  || true)
if [[ -n "$SECTION_LEAKS" ]]; then
  echo "FAIL: internal section IDs (S<phase>.<num>) present in tracked files:"
  echo "$SECTION_LEAKS" | sed 's/^/    /'
  exit 1
fi

printf '\n\033[1;32m✓ preflight OK\033[0m\n'
