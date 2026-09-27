#!/bin/bash
# Everything downstream of ingest: features -> train -> report -> demo -> bench.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
echo "=== M3 features ==="; $PY -u scripts/m3_features.py || exit 1
echo "=== M4 train ===";    $PY -u scripts/m4_train.py   || exit 1
echo "=== report ===";      $PY -u scripts/make_report.py >/dev/null && echo "reports/evaluation.md"
echo "=== M6 demo ===";     $PY -u scripts/m6_demo.py && $PY -u scripts/render_demo.py
echo "=== API latency ==="; $PY -u scripts/bench_api.py 120
