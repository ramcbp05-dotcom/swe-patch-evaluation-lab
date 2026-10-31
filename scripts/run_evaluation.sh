#!/usr/bin/env bash
# Full evaluation: every case, every declared candidate.
#
#   ./scripts/run_evaluation.sh
#
# Exits non-zero if any run is miscalibrated: a case that stops reproducing
# its own bug, a gold patch that fails, or a wrong patch that passes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$REPO_ROOT"
mkdir -p results

echo "== patches up to date? =="
"$PYTHON" scripts/build_patches.py --check

echo
echo "== library suite (the pass-to-pass baseline) =="
"$PYTHON" -m pytest tests/ -q

echo
echo "== lab integrity =="
"$PYTHON" -m pytest lab_tests/ -q

echo
echo "== evaluation =="
"$PYTHON" -m evaluation.cli evaluate \
    --report results/evaluation.json \
    --keep-logs results/logs \
    | tee results/evaluation.log
