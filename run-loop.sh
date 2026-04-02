#!/bin/bash

MAX_TURNS=50

if [ "$1" = "--dry-run" ]; then
  MAX_TURNS=1
  echo "Dry run mode: single turn per session"
fi

while true; do
  echo "=== Run started: $(date) ==="
  claude --dangerously-skip-permissions \
    --max-turns "$MAX_TURNS" \
    --verbose \
    -p "Read program.md in the current repo and start with your next hypothesis"
  echo "=== Run ended: $(date) ==="
done
