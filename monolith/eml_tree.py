import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from monolith.eml_ops import safe_eml
from monolith.symbolic import to_symbolic as _to_symbolic, snap_symbolic as _snap_symbolic


class EMLTree(nn.Module):
    """Complete binary tree where every node computes eml(L, R) = exp(L) - ln(R).

    Leaves are softmax mixtures over a candidate set [c0, c1, ..., x1, ..., xn].
    Trainable via gradient descent for symbolic regression.
    """

    def __init__(self, depth: int, n_vars: int = 1,
                 constants: tuple[float, ...] = (0.0, 1.0)) -> None:
        super().__init__()
        if depth < 1 or n_vars < 1:
            raise ValueError(f"depth={depth}, n_vars={n_vars}: both must be >= 1")

        self.depth = depth
        self.n_vars = n_vars
        self.constants = constants
        self.num_leaves = 2 ** depth
        self.num_candidates = len(constants) + n_vars

        self.leaf_logits = nn.Parameter(
            torch.randn(self.num_leaves, self.num_candidates) * 0.1
        )

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]

        const_cols = [torch.full((batch, 1), c, device=x.device, dtype=x.dtype)
                      for c in self.constants]
        candidates = torch.cat(const_cols + [x], dim=1)

        weights = F.softmax(self.leaf_logits, dim=-1)
        values = candidates @ weights.t()

        for _ in range(self.depth):
            values = safe_eml(values[..., 0::2], values[..., 1::2])

        return values.squeeze(-1)

    def to_symbolic(self) -> "sympy.Expr":
        """Decompile into a SymPy expression. High-confidence leaves snap to
        discrete candidates; others become linear combinations."""
        return _to_symbolic(self)

    def snap_symbolic(self, x_data: Tensor, y_data: Tensor,
                      tol: float = 2.0) -> "sympy.Expr | None":
        """Force-snap all leaves to argmax and verify against data.
        Returns the symbolic expression if MSE_snap/MSE_orig < tol, else None."""
        return _snap_symbolic(self, x_data, y_data, tol)

    def leaf_entropy(self) -> Tensor:
        """Per-leaf Shannon entropy of the softmax distribution."""
        p = F.softmax(self.leaf_logits, dim=-1)
        return -(p * p.log()).sum(dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _train_one(x, y, depth, n_vars, lr, grad_clip, epochs, seed,
                   init_logits=None, constants=(0.0, 1.0)):
        torch.manual_seed(seed)
        tree = EMLTree(depth=depth, n_vars=n_vars, constants=constants)

        if init_logits is not None:
            with torch.no_grad():
                n = init_logits.shape[0]
                tree.leaf_logits[:n] = init_logits.clone()

        opt = torch.optim.Adam(tree.parameters(), lr=lr)
        best, stale = float("inf"), 0

        for _ in range(epochs):
            loss = F.mse_loss(tree(x), y)
            if not torch.isfinite(loss):
                return tree, float("inf")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tree.parameters(), max_norm=grad_clip)
            opt.step()

            val = loss.item()
            if val < best - 1e-8:
                best, stale = val, 0
            else:
                stale += 1
                if stale >= 300:
                    break

        return tree, F.mse_loss(tree(x), y).item()

    @staticmethod
    def fit(x: Tensor, y: Tensor, max_depth: int = 3, n_restarts: int = 10,
            epochs: int = 10000, constants: tuple[float, ...] = (0.0, 1.0),
            verbose: bool = False) -> "EMLTree":
        """Hierarchical multi-depth search with per-depth hyperparameters.

        Depths 1-3 use independent random init. Depth 4+ warm-starts from the
        best tree at the previous depth.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        n_vars = x.shape[1]

        hp = {1: (0.01, 1.0), 2: (0.01, 1.0), 3: (0.001, 0.5)}
        hp_deep = (0.0005, 0.1)

        best_tree, best_mse = None, float("inf")
        prev_best = None

        for depth in range(1, max_depth + 1):
            lr, gc = hp.get(depth, hp_deep)
            warm = depth >= 4 and prev_best is not None

            depth_best_tree, depth_best_mse = None, float("inf")
            stale_seeds = 0

            for seed in range(n_restarts):
                init = prev_best.leaf_logits.detach() if warm else None
                tree, mse = EMLTree._train_one(
                    x, y, depth, n_vars, lr, gc, epochs, seed, init, constants)

                if mse < depth_best_mse:
                    depth_best_mse, depth_best_tree = mse, tree
                    stale_seeds = 0
                else:
                    stale_seeds += 1

                if mse < best_mse:
                    best_mse, best_tree = mse, tree
                    if verbose:
                        w = " (warm)" if warm else ""
                        print(f"  depth={depth} seed={seed}{w}: MSE={mse:.6f}")

                if stale_seeds >= 5 or best_mse < 1e-6:
                    break

            prev_best = depth_best_tree
            if best_mse < 1e-6:
                break

        if verbose:
            print(f"Best: depth={best_tree.depth}, MSE={best_mse:.6f}")
        return best_tree
