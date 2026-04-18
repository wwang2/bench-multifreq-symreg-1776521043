"""Closed-form fit for the multi-frequency symbolic regression benchmark.

Discovery procedure (see log.md for details):
    1. Stage-1 fit of basis [sin(x), cos(x), x^2, x, 1] via np.linalg.lstsq
       -> A_sin=1.005, A_cos=0.018, a=0.1050, b~0, c~-0.03
       -> dominant low-freq component is sin(x) with unit amplitude and
          a quadratic envelope x^2/10; train RMSE ~0.22 (too high, structure
          left in the residuals).
    2. np.fft.rfft of residuals on the uniform x grid reveals a strong peak
       straddling bins 6 & 7 (omega in [4.63, 5.41]), i.e. the true frequency
       sits exactly at omega=5 where FFT bin leakage splits energy between
       neighbours. No other peaks survive above noise.
    3. Stage-2 fit of basis [sin(x), cos(x), sin(5x), cos(5x), x^2, x, 1]
       -> A1_sin=1.006, A1_cos=0.018 (phi ~1 deg, ~zero)
          A2_sin=0.303, A2_cos=0.002 (phi ~0.4 deg, ~zero)
          a=0.1049, b~0, c~-0.03
       -> train RMSE 0.042, consistent with noise sigma=0.05.
    4. Coefficients snap cleanly to simple fractions (1.0, 0.3, 0.1) with
       zero phases and zero linear/constant terms.

Final closed-form:
    f(x) = sin(x) + 0.3 * sin(5*x) + 0.1 * x^2
"""
import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(x) + 0.3 * np.sin(5.0 * x) + 0.1 * x**2
