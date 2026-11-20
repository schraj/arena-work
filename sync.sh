#!/usr/bin/env bash
# Copy YOUR work out of the ARENA_materials clone into this showcase repo.
#
# "Your work" = whatever differs from upstream. Files you never touched stay upstream's,
# so they never land here — that is the whole point: this repo is your diff, not a copy
# of someone else's course.
set -euo pipefail

ARENA="${ARENA_DIR:-$HOME/code/ARENA_materials}"
BASE="${ARENA_BASE:-main}"   # clean mirror of upstream; never commit on it
DEST="$(cd "$(dirname "$0")" && pwd)"
[ -d "$ARENA/.git" ] || { echo "No ARENA clone at $ARENA (set ARENA_DIR)"; exit 1; }

cd "$ARENA"
FILES=$( { git diff --name-only --diff-filter=d "$BASE"; git ls-files --others --exclude-standard; } \
  | sort -u \
  | grep -E '^chapter[0-9]+_[^/]+/exercises/|^(requirements-macos\.txt$|EXERCISE_ISSUES\.md$|runpod/|scripts/)' \
  | grep -vE '(^|/)solutions\.py$|_solutions\.ipynb$' \
  | grep -vE '__pycache__|\.pyc$|/wandb/|\.pt$|\.pth$|\.ckpt$' || true )

[ -n "$FILES" ] || { echo "Nothing changed under chapter*/exercises or tooling yet — do some exercises first."; exit 0; }

echo "$FILES" | while read -r f; do echo "  + $f"; done
echo "$FILES" | rsync -R --files-from=- ./ "$DEST/"

cd "$DEST"
# Published copies only: don't leak the local username via saved outputs/tracebacks.
echo "$FILES" | while read -r f; do HOME_DIR="$HOME" perl -pi -e 's/\Q$ENV{HOME_DIR}\E/~/g' "$f"; done
python3 scripts/scan_secrets.py $(echo "$FILES" | tr '\n' ' ')
echo
echo "Synced. Review with 'git -C $DEST status', then commit."
