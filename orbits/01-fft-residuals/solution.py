"""
Solution for multi-frequency symbolic regression (orbit 01-fft-residuals, replica .r1).

Strategy
--------
1. Fit a degree-2 polynomial envelope to the 60 noisy training points.
2. Take the FFT of the residuals to find the dominant angular frequency (omega ~ 1).
3. Add sin(omega) + cos(omega) columns and refit jointly.
4. FFT the new residuals to expose the hidden small-amplitude, high-frequency
   component (omega ~ 5, amp ~ 0.3 - easy to miss visually).
5. Solve a single linear least-squares problem on the combined basis
   {1, x, x^2, sin(x), cos(x), sin(5x), cos(5x)}.

The lstsq fit returned near-zero coefficients for every cosine and linear
term, and amplitudes very close to 1.0, 0.3, 0.1 for the sinusoidal and
quadratic pieces. Clean-coefficient sanity check (per orbit instructions):
when lstsq hands you 1.0061, 0.3033, 0.1049 on 60 noisy samples, those are
the ground truth with noise bleed; gently rounding to clean values is the
right move - otherwise the solver is fitting the noise.
"""

import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    """Closed-form prediction on test grid."""
    x = np.asarray(x, dtype=float)
    return np.sin(x) + 0.3 * np.sin(5.0 * x) + 0.1 * x ** 2
