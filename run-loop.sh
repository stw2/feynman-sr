#!/bin/bash
# Autonomous research loop: one branch per investigation, benchmark-gated merge.
#
# Each iteration:
#   1. Reset to main
#   2. Claude Code runs one investigation (creates branch, publishes nanopub)
#   3. Benchmark the result branch
#   4. If improved → merge to main; otherwise → push branch only
set -euo pipefail
cd "$(dirname "$0")"

SESSIONS_DIR="${SESSIONS_DIR:-sessions}"
mkdir -p "$SESSIONS_DIR"

MAX_TURNS="${MAX_TURNS:-50}"
MAIN_BRANCH="main"

if [ "${1:-}" = "--dry-run" ]; then
  MAX_TURNS=1
  echo "Dry run mode: single turn per session"
fi

run_benchmark() {
  ./run.sh --all 2>&1 | grep '^summary_exact:' | awk '{print $2}'
}

# Measure baseline once
git checkout "$MAIN_BRANCH" --quiet
git pull --quiet origin "$MAIN_BRANCH" 2>/dev/null || true
BASELINE=$(run_benchmark)
BASELINE=${BASELINE:-0}
echo "==> Baseline: $BASELINE exact matches"

ITERATION=1
while true; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="$SESSIONS_DIR/run_${ITERATION}_${TIMESTAMP}.jsonl"

  echo ""
  echo "=== Iteration $ITERATION — $(date) ==="

  git checkout "$MAIN_BRANCH" --quiet
  git pull --quiet origin "$MAIN_BRANCH" 2>/dev/null || true

  claude --dangerously-skip-permissions \
    --max-turns "$MAX_TURNS" \
    --output-format stream-json \
    --verbose \
    -p "Read program.md in the current repo and start with your next hypothesis" \
    2>&1 | tee "$LOG_FILE"

  BRANCH=$(git branch --show-current)

  if [ "$BRANCH" = "$MAIN_BRANCH" ]; then
    echo "==> Agent stayed on main. Skipping."
    ITERATION=$((ITERATION + 1))
    continue
  fi

  RESULT=$(run_benchmark)
  RESULT=${RESULT:-0}
  echo "==> $BRANCH: $RESULT exact matches (baseline: $BASELINE)"

  if [ "$RESULT" -gt "$BASELINE" ]; then
    echo "==> Improvement. Merging $BRANCH to $MAIN_BRANCH."
    git checkout "$MAIN_BRANCH" --quiet
    git merge "$BRANCH" --no-ff -m "merge: $BRANCH ($BASELINE→$RESULT exact matches)"
    git push origin "$MAIN_BRANCH"
    BASELINE=$RESULT
  else
    echo "==> No improvement. Pushing branch only."
    git push -u origin "$BRANCH" 2>/dev/null || true
  fi

  ITERATION=$((ITERATION + 1))
done
