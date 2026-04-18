#!/usr/bin/env bash
# Reproduce orbit 01-fft-residuals: analysis, figures, eval.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

echo "[1/3] Stage-1/Stage-2 lstsq + FFT discovery ..."
python3 "${HERE}/explore.py"

echo "[2/3] Generate figures ..."
python3 "${HERE}/make_figures.py"

echo "[3/3] Evaluate across 3 seeds ..."
for SEED in 1 2 3; do
    python3 "${ROOT}/research/eval/evaluator.py" \
        --solution "${HERE}/solution.py" \
        --seed "${SEED}"
done
