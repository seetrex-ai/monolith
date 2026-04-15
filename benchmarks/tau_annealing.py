"""Benchmark: tau annealing vs baseline training.

Compares 7 elementary functions with and without tau annealing.
Reports MSE, mean leaf entropy, and snap_symbolic success rate.

Usage: python -u benchmarks/tau_annealing.py
"""

import sys
import time
import torch
import torch.nn.functional as F
from monolith import EMLTree

TARGETS = [
    ("exp(x)", lambda x: torch.exp(x), -2, 2),
    ("ln(x)", lambda x: torch.log(x), 0.5, 3),
    ("sqrt(x)", lambda x: torch.sqrt(x), 0.5, 4),
    ("x^2", lambda x: x**2, 0.5, 2.5),
    ("x^3", lambda x: x**3, 0.5, 2),
    ("1/x", lambda x: 1 / x, 0.5, 3),
    ("sin(x)", lambda x: torch.sin(x), -3, 3),
]

N_POINTS = 200
MAX_DEPTH = 3
N_RESTARTS = 20
EPOCHS = 10000


def evaluate(tree, x, y):
    mse = F.mse_loss(tree(x), y).item()
    entropy = tree.leaf_entropy().mean().item()
    snap = tree.snap_symbolic(x, y)
    return mse, entropy, snap


def main():
    print("=" * 100)
    print("BENCHMARK: Tau Annealing Effect on Discrete Recovery")
    print("=" * 100)
    print(f"Config: depth<={MAX_DEPTH}, restarts={N_RESTARTS}, epochs={EPOCHS}")
    print(f"Tau annealing: tau_search=2.5, tau_hard=0.01, hardening=25%, lam_entropy=0.02")
    print()

    header = (f"{'Function':10s} | {'Method':12s} | {'MSE':>12s} | "
              f"{'Entropy':>8s} | {'Snap':>5s} | {'Time':>7s} | Formula")
    print(header)
    print("-" * 100)

    baseline_snaps = 0
    tau_snaps = 0

    for name, fn_torch, lo, hi in TARGETS:
        x = torch.linspace(lo, hi, N_POINTS).unsqueeze(1)
        y = fn_torch(x.squeeze(1))

        # Baseline (no tau annealing)
        t0 = time.time()
        tree_base = EMLTree.fit(x, y, max_depth=MAX_DEPTH,
                                n_restarts=N_RESTARTS, epochs=EPOCHS)
        time_base = time.time() - t0
        mse_b, ent_b, snap_b = evaluate(tree_base, x, y)
        snap_str_b = str(snap_b) if snap_b is not None else "-"
        if snap_b is not None:
            baseline_snaps += 1
        print(f"{name:10s} | {'baseline':12s} | {mse_b:12.6f} | "
              f"{ent_b:8.4f} | {'YES' if snap_b else 'no':>5s} | "
              f"{time_base:6.1f}s | {snap_str_b}")
        sys.stdout.flush()

        # Tau annealing
        t0 = time.time()
        tree_tau = EMLTree.fit(x, y, max_depth=MAX_DEPTH,
                               n_restarts=N_RESTARTS, epochs=EPOCHS,
                               tau_search=2.5, tau_hard=0.01)
        time_tau = time.time() - t0
        mse_t, ent_t, snap_t = evaluate(tree_tau, x, y)
        snap_str_t = str(snap_t) if snap_t is not None else "-"
        if snap_t is not None:
            tau_snaps += 1
        print(f"{'':10s} | {'tau_anneal':12s} | {mse_t:12.6f} | "
              f"{ent_t:8.4f} | {'YES' if snap_t else 'no':>5s} | "
              f"{time_tau:6.1f}s | {snap_str_t}")
        sys.stdout.flush()
        print()

    print("=" * 100)
    print(f"SUMMARY: Snap success rate — baseline: {baseline_snaps}/{len(TARGETS)}, "
          f"tau_anneal: {tau_snaps}/{len(TARGETS)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
