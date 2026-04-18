# Multi-frequency symbolic regression

## Problem Statement
Fit a closed-form function f(x) to 60 noisy (x, y) training points on x ∈ [-4, 4]
(noise σ=0.05). The underlying function is expected to contain multiple
frequency components with some small-amplitude terms that are easy to miss from
a naive visual fit. Training data: `research/eval/train_data.csv`.

Constraints:
- NO sklearn, NO scipy.optimize, NO curve_fit.
- Coefficients must be tuned by inspection (FFT of residuals is OK for finding
  frequencies).
- Evaluator at `research/eval/evaluator.py` — do NOT rebuild it.

## Solution Interface
Solution must be a Python file at `orbits/<orbit>/solution.py` exporting
`f(x: np.ndarray) -> np.ndarray`. The evaluator calls `f` on the held-out
clean test grid.

## Success Metric
MSE on held-out clean test set (minimize). Target: MSE < 0.005.

## Budget
max_orbits = 3. Each orbit spawns 2 cross-validation replicas
(`execution.parallel_agents = 2`) that propose independently.
