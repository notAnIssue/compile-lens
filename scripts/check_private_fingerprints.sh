#!/usr/bin/env bash
#
# Pre-commit hook: refuse to commit files containing identifiers from internal
# planning notes that do not ship in this repository.
#
# Two pattern families are forbidden:
#
#   1. Paths into the maintainer's planning tree:
#        colibri/ , assets/notes/
#      A reader of this repository cannot follow those paths; they leak the
#      structure of work that is intentionally private.
#
#   2. Section identifiers of the form S<phase>.<num> :
#        S0.6  S0.16  S1.4  S7.X  ...
#      These index into internal todo files (one per phase). They look
#      authoritative but resolve to nothing a contributor can read.
#
# Both patterns appeared widely in phase 0 artifacts and required two
# retroactive scrub PRs (#13, #19). The rule that supersedes them, and the
# weighted matrix behind it, is in ADR-028.
#
# Whitelist: the files that exist *to describe these patterns* are exempted,
# because forbidding them from naming the patterns would make the rule
# unteachable. The exemption is by exact path, not by glob.

set -euo pipefail

WHITELIST=(
  "scripts/check_private_fingerprints.sh"
  "scripts/preflight.sh"
  ".pre-commit-config.yaml"
  "CONTRIBUTING.md"
  "docs/02_design_decisions/adr-028-naming-and-delivery.md"
)

is_whitelisted() {
  local f="$1"
  for w in "${WHITELIST[@]}"; do
    [[ "$f" == "$w" ]] && return 0
  done
  return 1
}

declare -i failures=0

for file in "$@"; do
  if is_whitelisted "$file"; then
    continue
  fi

  # Skip non-text or missing files (pre-commit can pass deleted paths).
  [[ -f "$file" ]] || continue

  # Pattern 1: planning-tree path references.
  if grep -nE 'colibri/|assets/notes/' "$file" >/dev/null 2>&1; then
    echo "FAIL $file : internal planning path (colibri/ or assets/notes/)"
    grep -nE 'colibri/|assets/notes/' "$file" | sed 's/^/    /'
    failures=$((failures + 1))
  fi

  # Pattern 2: S<phase>.<num> section IDs. Boundary set: start-of-line or
  # any non-alphanumeric on the left; non-alphanumeric or end-of-line on the
  # right. The trailing token is digits, X, or N (to also catch placeholders
  # like S7.X used during planning).
  if grep -nE '(^|[^A-Za-z0-9])S[0-9]+\.[0-9XN]+([^A-Za-z0-9]|$)' "$file" >/dev/null 2>&1; then
    echo "FAIL $file : internal section ID (S<phase>.<num>)"
    grep -nE '(^|[^A-Za-z0-9])S[0-9]+\.[0-9XN]+([^A-Za-z0-9]|$)' "$file" | sed 's/^/    /'
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  cat <<'EOF'

Refusing the commit. The patterns above leak identifiers that only resolve
inside the maintainer's planning tree. See CONTRIBUTING.md ("Naming") and
ADR-028 for the rule and the rewrite recipes.

If you genuinely need the pattern (e.g. you are editing one of the files
that exists to describe the rule), add the file to the WHITELIST in
scripts/check_private_fingerprints.sh.
EOF
  exit 1
fi
