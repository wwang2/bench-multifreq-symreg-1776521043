---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.0
---

# Research Notes — orbit `01-fft-residuals` (replica `.r1`)

## Result first
Fitted closed form on the 60 noisy samples:

$$f(x) = \sin(x) + 0.3\sin(5x) + 0.1\,x^2$$

Evaluator test MSE = **0.000000** across seeds 1, 2, 3 (below the 0.005 target
by four orders of magnitude). The test grid is a clean, deterministic
resampling of the same underlying function, so the rounded-to-clean
coefficients are exact — the metric is literally zero up to float round-off.

## Why the naive fit fails
A visual glance at `train_data.csv` suggests "sinusoid with a quadratic
envelope". Fitting just $\sin(x) + 0.1 x^2$ leaves a small oscillatory
residual of amplitude $\approx 0.3$, which is only $6\times$ the noise
standard deviation — easy to dismiss as noise (Figure `narrative.png(a)`).
The trick is to not trust the eye: **FFT the residuals** and see whether
any peak sticks out above the noise floor.

## Procedure

**Stage 1 — quadratic detrend.**
Fit $y \approx c_0 + c_1 x + c_2 x^2$ with `np.polyfit(x, y, 2)` and compute
residuals. Zero-pad the residual sequence to length 4096 and take a real
FFT. Convert cycles/unit to angular frequency ($\omega = 2\pi\,\nu$). A sharp
peak appears at $\omega \approx 1.17$ with amplitude $\approx 0.90$. The offset
from $\omega = 1$ is 60-sample spectral-leakage; a single sinusoid at
$\omega = 1$ is the cleanest explanation.

**Stage 2 — joint fit with $\sin(x), \cos(x)$.**
Add two columns and re-solve
$\min \lVert A\boldsymbol\beta - y \rVert_2^2$ with `np.linalg.lstsq`.
Residual std drops from 0.68 to 0.22. FFT the new residuals: a clean
secondary peak at $\omega \approx 5.02$ with amplitude $\approx 0.30$
emerges — exactly the small-amplitude, high-frequency term the instructions
warn is easy to miss.

**Stage 3 — joint fit with $\sin(5x), \cos(5x)$ added.**
Final basis $\{1, x, x^2, \sin x, \cos x, \sin 5x, \cos 5x\}$. The lstsq
solution returns:

| term      | fit       | clean |
|-----------|----------:|------:|
| const     | $-0.0314$ |  0.0  |
| x         | $-0.0007$ |  0.0  |
| $x^2$     | $+0.1049$ |  0.1  |
| $\sin(x)$ | $+1.0061$ |  1.0  |
| $\cos(x)$ | $+0.0178$ |  0.0  |
| $\sin(5x)$| $+0.3033$ |  0.3  |
| $\cos(5x)$| $+0.0021$ |  0.0  |

Every cosine and the linear coefficient land within noise of zero; every
surviving term lands within noise of a clean value ($0.1$, $1.0$, $0.3$).
Training MSE on the raw lstsq fit: $1.77 \times 10^{-3}$ (close to the
$\sigma^2 = 0.0025$ irreducible floor). Training MSE after gently rounding:
$2.19 \times 10^{-3}$ — a touch worse on the noisy points, but the right
move: the evaluator scores against the **clean** test function, not the
noisy samples.

Stage-3 residual FFT shows max amplitude 0.022 — indistinguishable from the
noise floor for 60 samples at $\sigma = 0.05$.

## Clean-coefficient sanity check
Per orbit instructions: "if lstsq returns amplitudes near 1.0, 0.3, 0.1, or
similar simple values, round gently — the ground truth likely has clean
coefficients and rounding helps avoid overfitting the 60 noisy samples."
All three dominant amplitudes are within half a percent of clean values;
all six "should be zero" coefficients are within noise of zero. Rounding is
the right call, and it was rewarded with MSE = 0.

## Results table
| Seed | Metric      | Time |
|------|-------------|------|
| 1    | 0.0000000000| <1s  |
| 2    | 0.0000000000| <1s  |
| 3    | 0.0000000000| <1s  |
| **Mean** | **0.0 ± 0.0** |  |

## Prior Art & Novelty

### What is already known
- The FFT-residual / iterative prewhitening procedure is standard practice
  in astronomical period-finding — e.g. CLEAN (Roberts, Lehár & Dreher,
  AJ 93, 968 (1987)) and its many descendants.
- Harmonic regression via a least-squares design matrix with sinusoid and
  polynomial columns is textbook (see e.g. Bloomfield, *Fourier Analysis
  of Time Series: An Introduction*, Wiley 2000).
- Spectral leakage correction and zero-padding for finer frequency
  resolution are basic DSP; see Harris (1978), "On the use of windows
  for harmonic analysis with the discrete Fourier transform".

### What this orbit adds
Nothing novel methodologically — this is a textbook application of the
above to a toy multi-frequency regression benchmark. The interesting bit
for the campaign is the **clean-coefficient rounding** step, which
exploits a prior that the benchmark was hand-constructed with simple
amplitudes; on real data this rounding would be unjustified.

### Honest positioning
The orbit confirms that the hypothesis "quadratic detrend + FFT residuals
+ joint lstsq" is enough to hit MSE = 0 on this benchmark when combined
with a modest rounding heuristic. No claim of generality beyond the
hand-crafted ground truth.

## What could still go wrong
- If the true function contained a cosine with comparable amplitude to the
  sines (breaking the "sin-only" prior), the rounding step would eliminate
  a real signal. The lstsq-without-rounding fallback still hits MSE =
  $3 \times 10^{-4}$, well below the 0.005 target.
- If the noise were non-Gaussian or heteroscedastic, the uniform-weight
  lstsq would be suboptimal; a weighted fit would help.
- For much higher frequencies ($\omega \gtrsim \pi / dx \approx 23$ here),
  aliasing from only 60 samples would make discovery impossible.

## Glossary
- **FFT** — Fast Fourier Transform.
- **lstsq** — least-squares solve, here via `np.linalg.lstsq`.
- **Spectral leakage** — spreading of a single-frequency signal across
  nearby FFT bins due to finite-length, non-integer-period sampling.
- **Prewhitening** — iteratively subtracting identified components from a
  residual and re-analysing; a.k.a. CLEAN in astronomy.
- **MSE** — mean squared error.

## References
- Roberts, Lehár & Dreher (1987) — "Time series analysis with CLEAN. I.
  Derivation of a spectrum". [ADS](https://ui.adsabs.harvard.edu/abs/1987AJ.....93..968R/abstract)
- Harris (1978) — "On the use of windows for harmonic analysis with the
  discrete Fourier transform". IEEE Proc. 66(1), 51–83.
- Bloomfield (2000) — *Fourier Analysis of Time Series: An Introduction*,
  2nd ed., Wiley.

## Files
- `solution.py` — closed-form f(x) exporter.
- `make_figures.py` — regenerates `figures/narrative.png` and `figures/results.png`.
- `figures/narrative.png` — baseline vs. method fits on the training data.
- `figures/results.png` — FFT waterfall across the three stages + coefficient
  table + seed MSE bars.
