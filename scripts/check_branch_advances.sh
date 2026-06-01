#!/usr/bin/env bash
#
# Pre-push hook: refuse to push a branch that has no commits beyond `main`.
#
# This catches the failure mode behind PR #23's wasted round-trip: a
# `git commit` was silently aborted by a pre-commit hook, the output was
# piped through `| tail -N` which swallowed the non-zero exit code, the
# subsequent `git push` happily uploaded the unchanged HEAD as a new
# "branch", and only `gh pr create` surfaced the problem with
#
#   GraphQL: No commits between main and <branch> (createPullRequest)
#
# The mechanical defense: at push time, assert that the local HEAD is at
# least one commit ahead of `main`. If it is not, refuse the push and tell
# the user where to look. Pushing `main` itself is always allowed.
#
# The cultural defense (don't pipe `git commit` through anything that
# swallows the exit code) is in CONTRIBUTING.md, "Pre-flight". This script
# is the belt to that suspenders.

set -euo pipefail

BASE_BRANCH="main"

current_branch=$(git rev-parse --abbrev-ref HEAD)

# Pushing main itself: no work to do.
if [[ "$current_branch" == "$BASE_BRANCH" ]]; then
  exit 0
fi

# A detached HEAD shouldn't be pushed in normal workflow, but if it is we
# don't have a sensible answer for "how many commits beyond main" — abstain.
if [[ "$current_branch" == "HEAD" ]]; then
  exit 0
fi

# If the base branch doesn't exist locally (fresh clone of a fork, weird
# remote setup), abstain rather than block the push.
if ! git rev-parse --verify --quiet "$BASE_BRANCH" >/dev/null; then
  exit 0
fi

count=$(git rev-list --count "${BASE_BRANCH}..HEAD")

if [[ "$count" -eq 0 ]]; then
  cat <<EOF >&2
REFUSE pushing $current_branch: no commits beyond $BASE_BRANCH.

This is the canary for "git commit was silently aborted but the push
happened anyway". Likely causes:

  - The previous commit command was piped through \`| tail\` or \`| head\`
    and the non-zero exit code from a pre-commit hook abort was swallowed.
  - A merge or rebase landed everything on $BASE_BRANCH but the working
    branch was never updated.

To diagnose: run \`git log --oneline ${BASE_BRANCH}..HEAD\` — it should
list at least one commit. If it does not, fix the missing commit and
push again.

See CONTRIBUTING.md "Pre-flight" for the rule that avoids piping
\`git commit\` through anything that swallows its exit code.
EOF
  exit 1
fi

exit 0
