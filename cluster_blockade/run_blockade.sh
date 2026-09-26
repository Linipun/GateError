#!/bin/bash
# One blockade job: ./run_blockade.sh <n> [extra blockade_asymmetric.py args]
# ARC caches matrix elements in $HOME/.arc-data (SQLite); parallel jobs sharing a
# home directory can lock/corrupt it, so each job gets its own HOME.
set -euo pipefail
N=$1; shift
export HOME="${TMPDIR:-$PWD}/arc_home_n${N}_$$"
mkdir -p "$HOME"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if ! python3 -c "import arc" 2>/dev/null; then
    python3 -m pip install --user --quiet "ARC-Alkali-Rydberg-Calculator==3.8.1"
fi

python3 blockade_asymmetric.py --n "$N" "$@"
rm -rf "$HOME"
