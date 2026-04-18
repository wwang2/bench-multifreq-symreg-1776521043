"""Analysis script: inspect data, fit stage 1 (sin + poly), FFT residuals, fit stage 2."""
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "research", "eval", "train_data.csv")

data = np.loadtxt(DATA, delimiter=",", skiprows=1)
x, y = data[:, 0], data[:, 1]
print(f"n={len(x)}, x in [{x.min():.3f}, {x.max():.3f}]")
dx = np.diff(x)
print(f"dx min/max: {dx.min():.6f} / {dx.max():.6f}  -> uniform? {np.allclose(dx, dx[0])}")

# Stage 1: sin(x), cos(x), x^2, x, 1 ----------------
B1 = np.column_stack([np.sin(x), np.cos(x), x**2, x, np.ones_like(x)])
c1, *_ = np.linalg.lstsq(B1, y, rcond=None)
yhat1 = B1 @ c1
res1 = y - yhat1
print(f"\nStage 1 coefficients [A_sin, A_cos, a(x^2), b(x), c]: {c1}")
print(f"Stage 1 train RMSE: {np.sqrt(np.mean(res1**2)):.5f}")

# FFT of residuals
n = len(x)
L = x[-1] - x[0]  # 8
fs = (n - 1) / L  # samples per unit x
freqs = np.fft.rfftfreq(n, d=L / (n - 1))  # cycles per unit x
F = np.fft.rfft(res1)
mag = np.abs(F)
# Convert to angular freq (ω in rad/unit)
omega = 2 * np.pi * freqs
# Top 5 peaks
order = np.argsort(mag)[::-1]
print("\nTop FFT peaks (omega, magnitude):")
for k in order[:8]:
    print(f"  k={k:2d}  cycles/unit={freqs[k]:.4f}  omega={omega[k]:.4f}  mag={mag[k]:.4f}")

# Candidate omega ~ 5
# Stage 2: add sin(w2*x), cos(w2*x) with w2=5
w2 = 5.0
B2 = np.column_stack([
    np.sin(x), np.cos(x),
    np.sin(w2 * x), np.cos(w2 * x),
    x**2, x, np.ones_like(x),
])
c2, *_ = np.linalg.lstsq(B2, y, rcond=None)
yhat2 = B2 @ c2
res2 = y - yhat2
print(f"\nStage 2 coefficients [A1s, A1c, A2s, A2c, a, b, c]: {c2}")
print(f"Stage 2 train RMSE: {np.sqrt(np.mean(res2**2)):.5f}  (noise sigma=0.05 -> expected ~0.05)")

# Check symbolic interpretation
A1 = np.hypot(c2[0], c2[1])
phi1 = np.arctan2(c2[1], c2[0])
A2 = np.hypot(c2[2], c2[3])
phi2 = np.arctan2(c2[3], c2[2])
print(f"  A1*sin(x + phi1): A1={A1:.4f}, phi1={phi1:.4f} rad ({np.degrees(phi1):.2f} deg)")
print(f"  A2*sin(5x + phi2): A2={A2:.4f}, phi2={phi2:.4f} rad ({np.degrees(phi2):.2f} deg)")

# Residuals FFT again
F2 = np.fft.rfft(res2)
mag2 = np.abs(F2)
order2 = np.argsort(mag2)[::-1]
print("\nStage 2 top FFT peaks:")
for k in order2[:5]:
    print(f"  omega={omega[k]:.4f}  mag={mag2[k]:.4f}")

# Try snapping to ground truth: sin(x) + 0.3*sin(5x) + 0.1*x^2
yhat_gt = np.sin(x) + 0.3 * np.sin(5 * x) + 0.1 * x**2
print(f"\nGround-truth candidate train RMSE: {np.sqrt(np.mean((y - yhat_gt)**2)):.5f}")
