#!/bin/bash
# Runs the remaining ingest stages back to back. Every stage is cached and
# resumable, so re-running after an interruption costs nothing.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
echo "=== M1b victims  $(date +%H:%M:%S) ==="
$PY -u scripts/m1b_victims.py "${1:-800}" "${2:-4}" 2>&1 | grep -vE "^INFO httpx"
echo "=== M2 controls  $(date +%H:%M:%S) ==="
$PY -u scripts/m2_controls.py "${3:-250}" "${4:-400}" "${5:-4}" 2>&1 | grep -vE "^INFO httpx"
echo "=== ingest chain complete $(date +%H:%M:%S) ==="
