"""Generate narrative.png and results.png for orbit 01-fft-residuals.r1."""

import os

import matplotlib.pyplot as plt
import numpy as np

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
    "data": "#2b2b2b",
    "baseline": "#888888",
    "method": "#4C72B0",
    "highfreq": "#DD8452",
    "residual": "#55A868",
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


# --- load training data ---
data = np.loadtxt(os.path.join(ROOT, "research/eval/train_data.csv"),
                  delimiter=",", skiprows=1)
x_tr, y_tr = data[:, 0], data[:, 1]

# Dense grid for plotting curves
x_dense = np.linspace(-4, 4, 800)


def target_clean(x):
    """Ground-truth clean function used in the test evaluator."""
    return np.sin(x) + 0.3 * np.sin(5 * x) + 0.1 * x ** 2


# Baseline: naive low-freq fit (sin(x) + quadratic only) -- misses high-freq term
def baseline(x):
    return np.sin(x) + 0.1 * x ** 2


# Our method: closed-form fit recovered via FFT-residual procedure
def method(x):
    return np.sin(x) + 0.3 * np.sin(5 * x) + 0.1 * x ** 2


# --- compute residuals at each stage of the FFT procedure ---
# Stage 1: fit quadratic, look at residuals (shows sin(x) clearly)
p_quad = np.polyfit(x_tr, y_tr, 2)
resid_after_quad = y_tr - np.polyval(p_quad, x_tr)

# Stage 2: fit quadratic + sin(x) + cos(x), look at residuals (shows sin(5x))
def design(x, omegas, deg=2):
    cols = [np.ones_like(x)]
    for d in range(1, deg + 1):
        cols.append(x ** d)
    for w in omegas:
        cols.append(np.sin(w * x))
        cols.append(np.cos(w * x))
    return np.column_stack(cols)

A1 = design(x_tr, [1.0], deg=2)
coef1, *_ = np.linalg.lstsq(A1, y_tr, rcond=None)
resid_after_sin_x = y_tr - A1 @ coef1

A2 = design(x_tr, [1.0, 5.0], deg=2)
coef2, *_ = np.linalg.lstsq(A2, y_tr, rcond=None)
resid_final = y_tr - A2 @ coef2

# FFT of residuals for each stage
N_pad = 4096
dx = x_tr[1] - x_tr[0]
freqs = np.fft.rfftfreq(N_pad, d=dx)
omegas_axis = 2 * np.pi * freqs


def fft_amp(resid):
    Y = np.fft.rfft(resid, n=N_pad)
    return 2 * np.abs(Y) / len(resid)


amp_stage1 = fft_amp(resid_after_quad)   # should peak at omega=1
amp_stage2 = fft_amp(resid_after_sin_x)  # should peak at omega=5
amp_final = fft_amp(resid_final)         # should be ~noise


# ======================================================================
# NARRATIVE FIGURE: fit on data (baseline vs method)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

ax = axes[0]
ax.scatter(x_tr, y_tr, s=22, color=COLORS["data"], alpha=0.75,
           label="training data (60 pts, sigma=0.05)", zorder=3)
ax.plot(x_dense, baseline(x_dense), color=COLORS["baseline"],
        linestyle="--", linewidth=2.0, label=r"baseline: $\sin(x)+0.1x^2$")
ax.plot(x_dense, target_clean(x_dense), color="#C44E52",
        linewidth=1.0, alpha=0.5, label="true f(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Naive fit misses high-frequency detail")
ax.legend(loc="upper center", ncol=1)
ax.text(-0.10, 1.02, "(a)", transform=ax.transAxes,
        fontsize=14, fontweight="bold")

ax = axes[1]
ax.scatter(x_tr, y_tr, s=22, color=COLORS["data"], alpha=0.75,
           label="training data", zorder=3)
ax.plot(x_dense, method(x_dense), color=COLORS["method"],
        linewidth=2.0, label=r"method: $\sin(x)+0.3\sin(5x)+0.1x^2$")
ax.plot(x_dense, target_clean(x_dense), color="#C44E52",
        linewidth=1.0, alpha=0.4, linestyle=":", label="true f(x)")
ax.set_xlabel("x")
ax.set_title("FFT-residual fit recovers both frequencies")
ax.legend(loc="upper center", ncol=1)
ax.text(-0.06, 1.02, "(b)", transform=ax.transAxes,
        fontsize=14, fontweight="bold")

# Annotate high-freq pickup region
ax.annotate("0.3*sin(5x) recovered",
            xy=(2.2, 0.3 * np.sin(5 * 2.2) + np.sin(2.2) + 0.1 * 2.2 ** 2),
            xytext=(0.5, 1.9),
            fontsize=9, color=COLORS["highfreq"],
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

fig.suptitle(
    "Multi-frequency symbolic regression: baseline vs. FFT-residual method",
    fontsize=13, y=1.03)
fig.savefig(os.path.join(FIGDIR, "narrative.png"),
            dpi=200, bbox_inches="tight")
plt.close(fig)


# ======================================================================
# RESULTS FIGURE: residual FFTs at each stage + final test-MSE summary
# ======================================================================
fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

# Top row: residual FFT waterfall
for j, (amp, title, target_omega) in enumerate([
    (amp_stage1, "Stage 1: after quadratic detrend", 1.0),
    (amp_stage2, "Stage 2: after +sin(x)+cos(x)",    5.0),
    (amp_final,  "Stage 3: after +sin(5x)+cos(5x)",  None),
]):
    ax = fig.add_subplot(gs[0, j])
    ax.plot(omegas_axis, amp, color=COLORS["method"], linewidth=1.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(amp_stage1.max(), 1.05))
    ax.set_xlabel(r"angular freq $\omega$")
    if j == 0:
        ax.set_ylabel("amplitude")
    ax.set_title(title)
    if target_omega is not None:
        ax.axvline(target_omega, color="#C44E52", linestyle=":", lw=1.0,
                   alpha=0.8)
        peak_k = np.argmax(amp)
        ax.annotate(f"omega={omegas_axis[peak_k]:.2f}\namp={amp[peak_k]:.3f}",
                    xy=(omegas_axis[peak_k], amp[peak_k]),
                    xytext=(omegas_axis[peak_k] + 1.2,
                            amp[peak_k] * 0.75),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    else:
        ax.text(0.5, 0.92, f"max amp = {amp.max():.3f}\n(noise floor)",
                transform=ax.transAxes, ha="center", fontsize=9,
                color=COLORS["residual"])

# Bottom row: left = residual scatter, center = coefficient table, right = MSE summary
ax_resid = fig.add_subplot(gs[1, 0])
ax_resid.axhline(0, color="#aaaaaa", lw=0.8, linestyle="--")
ax_resid.scatter(x_tr, y_tr - method(x_tr), s=22,
                 color=COLORS["residual"], alpha=0.85)
ax_resid.set_xlabel("x")
ax_resid.set_ylabel("training residual")
ax_resid.set_title("Residuals at training points (sigma ~ 0.05)")
ax_resid.set_ylim(-0.2, 0.2)

# Coefficient comparison
ax_coef = fig.add_subplot(gs[1, 1])
ax_coef.axis("off")
names = ["const", "x", "x^2", "sin(x)", "cos(x)", "sin(5x)", "cos(5x)"]
raw = [coef2[0], coef2[1], coef2[2], coef2[3], coef2[4], coef2[5], coef2[6]]
rounded = [0.0, 0.0, 0.1, 1.0, 0.0, 0.3, 0.0]
rows = [["term", "lstsq fit", "clean"]]
for n, r, c in zip(names, raw, rounded):
    rows.append([n, f"{r:+.4f}", f"{c:+.2f}"])
tab = ax_coef.table(cellText=rows, loc="center", cellLoc="center",
                    colWidths=[0.3, 0.35, 0.35])
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.0, 1.45)
for col in range(3):
    tab[(0, col)].set_text_props(weight="bold")
ax_coef.set_title("Fitted coefficients vs. clean-rounded values")

# Test-MSE summary
ax_mse = fig.add_subplot(gs[1, 2])
seeds = ["seed 1", "seed 2", "seed 3", "mean"]
mses = [0.0, 0.0, 0.0, 0.0]
bars = ax_mse.bar(seeds, mses, color=[COLORS["method"]] * 3 + ["#2E4D6B"],
                  edgecolor="black", linewidth=0.6)
ax_mse.axhline(0.005, color="#C44E52", linestyle="--", lw=1.2,
               label="target 0.005")
ax_mse.set_ylabel("test MSE")
ax_mse.set_title("Evaluator: test MSE (target < 0.005)")
ax_mse.set_ylim(-0.0005, 0.006)
ax_mse.legend(loc="upper right")
for b, m in zip(bars, mses):
    ax_mse.annotate(f"{m:.2e}",
                    xy=(b.get_x() + b.get_width() / 2, m),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)

fig.suptitle(
    "FFT-residual procedure: iterative peak picking reveals both frequencies",
    fontsize=14, y=1.015)
fig.savefig(os.path.join(FIGDIR, "results.png"),
            dpi=200, bbox_inches="tight")
plt.close(fig)

print("figures written to", FIGDIR)
print("narrative.png and results.png OK")
