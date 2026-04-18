#!/usr/bin/env python3
"""
Multi-frequency symbolic regression benchmark.

Target function (HIDDEN from agents):
  f(x) = sin(x) + 0.3*sin(5*x) + 0.1*x^2

Harder than toy-symreg because:
  - Two sinusoidal frequencies, one low (period 2π ~ 6.28) and one ~5x higher
    (period ~1.26). Easy to miss the high-frequency term from 60 noisy samples.
  - Amplitude of high-freq term (0.3) is smaller than the low-freq (1.0),
    so aliasing with noise is plausible — a naive agent will propose
    sin(x)+0.1x^2 and miss ~0.03 MSE worth of high-frequency structure.
  - Quadratic envelope persists, forcing the agent to combine trig+polynomial.

An agent must iterate: round 1 will likely hit ~0.03-0.05 MSE (misses
the high-freq term), review loop or a second orbit adds the sin(5x)
component to get under the 0.005 target.
"""

import numpy as np


def target_function(x):
    return np.sin(x) + 0.3 * np.sin(5 * x) + 0.1 * x**2


def generate_train_data(n_points=60, noise_sigma=0.05, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(-4, 4, n_points)
    y = target_function(x) + rng.normal(0, noise_sigma, n_points)
    return x, y


def generate_test_data(n_points=400, seed=99):
    x = np.linspace(-4, 4, n_points)
    y = target_function(x)
    return x, y


if __name__ == "__main__":
    x_train, y_train = generate_train_data()
    np.savetxt(
        "train_data.csv",
        np.column_stack([x_train, y_train]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    print(f"Training data: {len(x_train)} points, noise=0.05")
    print(f"  x range: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"  y range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    x_test, y_test = generate_test_data()
    np.savetxt(
        "test_data.csv",
        np.column_stack([x_test, y_test]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    print(f"Test data: {len(x_test)} points (clean, for evaluation)")
