"""Benchmark Monolith vs PySR on identical functions.

Usage: PYTHONIOENCODING=utf-8 python -u benchmarks/vs_pysr.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

TARGETS = [
    ("exp(x)", lambda x: np.exp(x), lambda x: torch.exp(x), -2, 2),
    ("ln(x)", lambda x: np.log(x), lambda x: torch.log(x), 0.5, 3),
    ("sqrt(x)", lambda x: np.sqrt(x), lambda x: torch.sqrt(x), 0.5, 4),
    ("x^2", lambda x: x**2, lambda x: x**2, 0.5, 2.5),
    ("x^3", lambda x: x**3, lambda x: x**3, 0.5, 2),
    ("1/x", lambda x: 1 / x, lambda x: 1 / x, 0.5, 3),
    ("sin(x)", lambda x: np.sin(x), lambda x: torch.sin(x), -3, 3),
    ("sin(x^2)", lambda x: np.sin(x**2), lambda x: torch.sin(x**2), 0, 2.5),
    ("exp(sin(x))", lambda x: np.exp(np.sin(x)), lambda x: torch.exp(torch.sin(x)), -3, 3),
]

N_POINTS = 200


def run_monolith(fn_torch, lo, hi):
    from monolith import EMLTree

    x = torch.linspace(lo, hi, N_POINTS).unsqueeze(1)
    y = fn_torch(x.squeeze(1))

    t0 = time.time()
    tree = EMLTree.fit(x, y, max_depth=5, n_restarts=20, epochs=10000)
    elapsed = time.time() - t0

    mse = F.mse_loss(tree(x), y).item()
    snap = tree.snap_symbolic(x, y)
    formula = str(snap) if snap is not None else "(continuous mixture)"
    return mse, elapsed, formula, tree.depth


def run_pysr(fn_np, lo, hi):
    from pysr import PySRRegressor

    x_np = np.linspace(lo, hi, N_POINTS).reshape(-1, 1)
    y_np = fn_np(x_np.ravel())

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "sin", "cos"],
        populations=15, population_size=33, maxsize=20,
        timeout_in_seconds=60, temp_equation_file=True, verbosity=0,
    )

    t0 = time.time()
    model.fit(x_np, y_np)
    elapsed = time.time() - t0

    y_pred = model.predict(x_np)
    mse = float(np.mean((y_pred - y_np) ** 2))
    return mse, elapsed, str(model.sympy())


def main():
    print("=" * 90)
    print("BENCHMARK: Monolith EMLTree vs PySR")
    print("=" * 90)
    print(f"{'Function':15s} | {'Method':8s} | {'MSE':>12s} | {'Time':>8s} | {'Formula'}")
    print("-" * 90)

    for name, fn_np, fn_torch, lo, hi in TARGETS:
        mse_m, time_m, formula_m, depth_m = run_monolith(fn_torch, lo, hi)
        print(f"{name:15s} | {'Monolith':8s} | {mse_m:12.6f} | {time_m:7.1f}s | d={depth_m} {formula_m}")
        sys.stdout.flush()

        try:
            mse_p, time_p, formula_p = run_pysr(fn_np, lo, hi)
            print(f"{'':15s} | {'PySR':8s} | {mse_p:12.6f} | {time_p:7.1f}s | {formula_p}")
        except Exception as e:
            print(f"{'':15s} | {'PySR':8s} | {'ERROR':>12s} | {'':>8s} | {e}")
        sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
