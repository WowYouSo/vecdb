#!/usr/bin/env bash
# Reproduce every number: tests, all benchmarks, plots, summary.
#   ./run_all.sh              full SIFT1M (needs data/sift-128-euclidean.hdf5)
#   N=200000 ./run_all.sh     quick pass on a subset
set -euo pipefail
cd "$(dirname "$0")"
N=${N:-1000000}
NLIST=${NLIST:-4096}
PY=${PY:-python}

$PY -m pytest -q
$PY bench/bench_search.py --n "$N"
$PY bench/bench_faiss.py --n "$N"
$PY bench/bench_writes.py --n "$N" --nlist "$NLIST"
$PY bench/crash_test.py
$PY bench/bench_delta.py --n "$N" --nlist "$NLIST"
$PY bench/report.py
echo "Done: open results/summary.md"
