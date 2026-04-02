#!/bin/bash
# Run symbolic regression and capture output to run.log.
# Usage: ./run.sh --equation eq01 [--equation eq06 ...]
#        ./run.sh --all
#        ./run.sh --tier 1
set -euo pipefail
cd "$(dirname "$0")"

# Use venv python if available
if [ -f .venv/bin/python ]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

# Ensure equation data + metadata exist (prepare.py has ground truth — not for agents)
if [ ! -f equations/metadata.json ]; then
  echo "Generating equation data..."
  $PYTHON prepare.py
fi

# Run GP
$PYTHON train.py "$@" 2>&1 | tee run.log

echo ""
echo "Output saved to run.log"
