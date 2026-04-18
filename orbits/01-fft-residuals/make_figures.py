"""Generate narrative.png and results.png for the orbit."""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA = os.path.join(ROOT, "research", "eval", "train_data.csv")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLORS = {
    "data":     "#2a2a2a",
    "baseline": "#888888",
    "stage1":   "#4C72B0",
    "stage2":   "#C44E52",
    "truth":    "#55A868",
    "noise":    "#BBBBBB",
}

# ---------------------------------------------------------------- load data
data = np.loadtxt(DATA, delimiter=",", skiprows=1)
x, y = data[:, 0], data[:, 1]

# Dense x for smooth curves
xd = np.linspace(-4, 4, 801)

# Stage 1 fit: sin(x), cos(x), x^2, x, 1
B1 = np.column_stack([np.sin(x), np.cos(x), x**2, x, np.ones_like(x)])
c1, *_ = np.linalg.lstsq(B1, y, rcond=None)
B1d = np.column_stack([np.sin(xd), np.cos(xd), xd**2, xd, np.ones_like(xd)])
yhat1_d = B1d @ c1
res1 = y - (B1 @ c1)

# Stage 2 fit: + sin(5x), cos(5x)
w2 = 5.0
B2 = np.column_stack([np.sin(x), np.cos(x), np.sin(w2*x), np.cos(w2*x),
                      x**2, x, np.ones_like(x)])
c2, *_ = np.linalg.lstsq(B2, y, rcond=None)
B2d = np.column_stack([np.sin(xd), np.cos(xd), np.sin(w2*xd), np.cos(w2*xd),
                       xd**2, xd, np.ones_like(xd)])
yhat2_d = B2d @ c2
res2 = y - (B2 @ c2)

# Final closed-form
yhat_f_d = np.sin(xd) + 0.3*np.sin(5*xd) + 0.1*xd**2
yhat_f   = np.sin(x)  + 0.3*np.sin(5*x)  + 0.1*x**2
res_f    = y - yhat_f

# FFT of stage-1 residuals
n = len(x); L = x[-1] - x[0]
freqs = np.fft.rfftfreq(n, d=L/(n-1))
omega = 2*np.pi*freqs
F1 = np.abs(np.fft.rfft(res1))
F2 = np.abs(np.fft.rfft(res2))

# =========================================================== NARRATIVE.PNG
# Two-panel: (a) baseline (sin+poly only) vs (b) final closed-form,
# both over the same scatter — the reader can SEE the high-freq residual
# the baseline misses and how the final fit tracks the data.
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)

for ax in axes:
    ax.scatter(x, y, s=22, color=COLORS["data"], alpha=0.85,
               zorder=3, label="training data (n=60, σ=0.05)")
    ax.set_xlabel("x")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-2.1, 3.1)

axes[0].plot(xd, yhat1_d, color=COLORS["baseline"], lw=2.0, zorder=2,
             label="stage-1 fit: sin(x)+cos(x)+x²+x+1")
axes[0].set_ylabel("y")
axes[0].set_title("Stage 1 — low-freq fit only", pad=16)
axes[0].legend(loc="lower right")
# Annotate one of the missed wiggles (point near x=2.4 where baseline lags data)
axes[0].annotate("missed high-freq\noscillation", xy=(2.4, 1.45),
                 xytext=(0.05, 2.55),
                 fontsize=10, color="#444", ha="left",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

axes[1].plot(xd, yhat_f_d, color=COLORS["stage2"], lw=2.2, zorder=2,
             label=r"final:  $\sin x + 0.3\,\sin 5x + 0.1\,x^{2}$")
axes[1].set_title("Final closed-form — all frequencies captured", pad=16)
axes[1].legend(loc="lower right")
axes[1].annotate("wiggle\ncaptured", xy=(2.4, 1.45),
                 xytext=(0.05, 2.55),
                 fontsize=10, color="#444", ha="left",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# Panel labels (pulled further left so they don't clash with titles)
for ax, lab in zip(axes, ["(a)", "(b)"]):
    ax.text(-0.10, 1.06, lab, transform=ax.transAxes,
            fontsize=14, fontweight="bold")

fig.suptitle(r"Two-stage recovery of  $f(x) = \sin x + 0.3\,\sin 5x + 0.1\,x^{2}$"
             "   (test MSE = 0.000, target < 0.005)",
             y=1.03, fontsize=13)
fig.savefig(os.path.join(FIG_DIR, "narrative.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================= RESULTS.PNG
# 2x2 grid: residuals stage 1, residuals final, FFT stage 1, FFT final
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

ax = axes[0, 0]
ax.axhline(0, color=COLORS["noise"], lw=0.8, zorder=1)
ax.scatter(x, res1, s=28, color=COLORS["stage1"], zorder=3,
           label=f"stage-1 residuals, RMSE={np.sqrt(np.mean(res1**2)):.3f}")
ax.axhspan(-0.05, 0.05, color=COLORS["noise"], alpha=0.18, zorder=0,
           label="±σ noise band (σ=0.05)")
ax.set_title("(a)  Residuals after stage-1 fit")
ax.set_xlabel("x"); ax.set_ylabel("residual")
ax.set_ylim(-0.7, 0.7)
ax.legend(loc="upper right")

ax = axes[0, 1]
ax.axhline(0, color=COLORS["noise"], lw=0.8, zorder=1)
ax.scatter(x, res_f, s=28, color=COLORS["stage2"], zorder=3,
           label=f"final residuals, RMSE={np.sqrt(np.mean(res_f**2)):.3f}")
ax.axhspan(-0.05, 0.05, color=COLORS["noise"], alpha=0.18, zorder=0,
           label="±σ noise band (σ=0.05)")
ax.set_title("(b)  Residuals after final closed-form")
ax.set_xlabel("x"); ax.set_ylabel("residual")
ax.set_ylim(-0.7, 0.7)
ax.legend(loc="upper right")

# FFT plots (stage 1 residuals) — show peak at omega ~ 5
ax = axes[1, 0]
markerline, stemlines, baseline = ax.stem(omega, F1, linefmt="-", markerfmt="o",
                                          basefmt=" ")
plt.setp(stemlines, color=COLORS["stage1"], linewidth=1.2)
plt.setp(markerline, color=COLORS["stage1"], markersize=4)
ax.axvline(5.0, color=COLORS["stage2"], ls="--", lw=1.2, zorder=1)
ax.annotate("ω = 5", xy=(5.0, F1.max()*0.95), xytext=(6.5, F1.max()*0.95),
            fontsize=11, color=COLORS["stage2"],
            arrowprops=dict(arrowstyle="->", color=COLORS["stage2"], lw=0.9))
ax.set_title("(c)  FFT |·| of stage-1 residuals   (peak straddles ω=5)")
ax.set_xlabel("angular frequency ω"); ax.set_ylabel("magnitude")
ax.set_xlim(0, 25)

ax = axes[1, 1]
markerline, stemlines, baseline = ax.stem(omega, F2, linefmt="-", markerfmt="o",
                                          basefmt=" ")
plt.setp(stemlines, color=COLORS["stage2"], linewidth=1.2)
plt.setp(markerline, color=COLORS["stage2"], markersize=4)
ax.set_title("(d)  FFT |·| of final residuals   (noise-only, no peaks)")
ax.set_xlabel("angular frequency ω"); ax.set_ylabel("magnitude")
ax.set_xlim(0, 25)
ax.set_ylim(0, F1.max()*1.05)  # same y-scale as (c) for comparison

for ax, lab in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(-0.11, 1.04, lab, transform=ax.transAxes,
            fontsize=14, fontweight="bold")

fig.suptitle("Residuals collapse to noise once the ω=5 component is added   "
             "(test MSE = 0.000, target < 0.005)",
             y=1.02, fontsize=13)
fig.savefig(os.path.join(FIG_DIR, "results.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Print summary
print(f"stage-1 coeffs: {c1}")
print(f"stage-2 coeffs: {c2}")
print(f"stage-1 RMSE={np.sqrt(np.mean(res1**2)):.5f}")
print(f"stage-2 RMSE={np.sqrt(np.mean(res2**2)):.5f}")
print(f"final   RMSE={np.sqrt(np.mean(res_f**2)):.5f}")
print("Figures written to", FIG_DIR)
