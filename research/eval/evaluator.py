#!/usr/bin/env python3
"""
Evaluator for the toy symbolic-regression benchmark.

Scores solution.py's f(x) against held-out clean test data.

Usage:
    python evaluator.py --solution <path> --seed <int>
    Output: METRIC=<float>  (MSE on test set)
"""

import argparse
import importlib.util
import os
import sys

import numpy as np


BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_DIR)
from generate_data import generate_test_data


def load_solution(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(solution_path, seed=42):
    try:
        module = load_solution(solution_path)
    except Exception as e:
        print(f"ERROR: Could not load solution: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(module, "f"):
        print("ERROR: solution.py must define f(x)", file=sys.stderr)
        sys.exit(1)
    predict = module.f

    x_test, y_test = generate_test_data()
    try:
        y_pred = np.asarray(predict(x_test), dtype=float)
    except Exception as e:
        print(f"ERROR: Solution prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

    if y_pred.shape != y_test.shape:
        print(
            f"ERROR: predicted shape {y_pred.shape} != test shape {y_test.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    mse = float(np.mean((y_pred - y_test) ** 2))
    print(f"METRIC={mse:.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.solution, args.seed)
