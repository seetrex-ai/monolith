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

    def forward(self, x: Tensor, tau: float = 1.0) -> Tensor:
        batch = x.shape[0]

        const_cols = [torch.full((batch, 1), c, device=x.device, dtype=x.dtype)
                      for c in self.constants]
        candidates = torch.cat(const_cols + [x], dim=1)

        weights = F.softmax(self.leaf_logits / tau, dim=-1)
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

    def leaf_entropy(self, tau: float = 1.0) -> Tensor:
        """Per-leaf Shannon entropy of the softmax distribution."""
        p = F.softmax(self.leaf_logits / tau, dim=-1)
        return -(p * p.log()).sum(dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _hard_project(tree):
        """Snap all leaf logits to argmax (one-hot with large magnitude)."""
        from copy import deepcopy
        snapped = deepcopy(tree)
        with torch.no_grad():
            idx = torch.argmax(snapped.leaf_logits, dim=-1)
            new_logits = torch.full_like(snapped.leaf_logits, -24.0)
            new_logits[torch.arange(snapped.num_leaves), idx] = 24.0
            snapped.leaf_logits.copy_(new_logits)
        return snapped

    @staticmethod
    def _train_one(x, y, depth, n_vars, lr, grad_clip, epochs, seed,
                   init_logits=None, constants=(0.0, 1.0),
                   tau_search=1.0, tau_hard=0.01,
                   hardening_fraction=0.25, hardening_tau_power=2.0,
                   lam_entropy=0.02):
        torch.manual_seed(seed)
        tree = EMLTree(depth=depth, n_vars=n_vars, constants=constants)

        if init_logits is not None:
            with torch.no_grad():
                n = init_logits.shape[0]
                tree.leaf_logits[:n] = init_logits.clone()

        use_annealing = tau_search > 1.0 + 1e-6
        hardening_epochs = int(epochs * hardening_fraction) if use_annealing else 0
        search_epochs = epochs - hardening_epochs

        opt = torch.optim.Adam(tree.parameters(), lr=lr)
        best, stale = float("inf"), 0
        phase = "search"
        hard_start = search_epochs
        best_state = None
        best_hard_state = None

        for epoch in range(epochs):
            # Transition: scheduled or plateau-triggered
            if use_annealing and phase == "search" and (
                epoch >= search_epochs or stale >= 300
            ):
                phase = "hardening"
                hard_start = epoch
                stale = 0
                # Restore best search state before hardening
                if best_state is not None:
                    tree.load_state_dict(best_state)
                    opt = torch.optim.Adam(tree.parameters(), lr=lr)

            if phase == "search":
                tau = tau_search if use_annealing else 1.0
                lam_ent = 0.0
            else:
                t = (epoch - hard_start) / max(1, hardening_epochs - 1)
                t = min(t, 1.0)
                t_tau = t ** hardening_tau_power
                tau = tau_search * (tau_hard / tau_search) ** t_tau
                lam_ent = t * lam_entropy

            pred = tree(x, tau=tau)
            mse = F.mse_loss(pred, y)

            if not torch.isfinite(mse):
                if use_annealing and best_hard_state is not None:
                    # NaN during hardening: restore last good state and continue
                    tree.load_state_dict(best_hard_state)
                    opt = torch.optim.Adam(tree.parameters(), lr=lr)
                    continue
                elif use_annealing:
                    continue
                return tree, float("inf"), None, float("inf")

            if lam_ent > 0:
                entropy = tree.leaf_entropy(tau=tau).mean()
                loss = mse + lam_ent * entropy
            else:
                loss = mse

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tree.parameters(), max_norm=grad_clip)
            opt.step()

            val = mse.item()
            if val < best - 1e-8:
                best, stale = val, 0
                state = {k: v.clone() for k, v in tree.state_dict().items()}
                if phase == "search":
                    best_state = state
                else:
                    best_hard_state = state
            else:
                stale += 1
                if not use_annealing and stale >= 300:
                    break

        # Restore best state for final evaluation
        if use_annealing:
            final_state = best_hard_state or best_state
            if final_state is not None:
                tree.load_state_dict(final_state)

        soft_mse = F.mse_loss(tree(x), y).item()

        if use_annealing:
            snapped = EMLTree._hard_project(tree)
            snap_mse = F.mse_loss(snapped(x), y).item()
            return tree, soft_mse, snapped, snap_mse

        return tree, soft_mse, None, float("inf")

    @staticmethod
    def fit(x: Tensor, y: Tensor, max_depth: int = 3, n_restarts: int = 10,
            epochs: int = 10000, constants: tuple[float, ...] = (0.0, 1.0),
            verbose: bool = False,
            tau_search: float = 1.0, tau_hard: float = 0.01,
            hardening_fraction: float = 0.25,
            lam_entropy: float = 0.02) -> "EMLTree":
        """Hierarchical multi-depth search with per-depth hyperparameters.

        Depths 1-3 use independent random init. Depth 4+ warm-starts from the
        best tree at the previous depth.

        Tau annealing (inspired by Odrzywołek, arXiv 2603.21852):
            Set tau_search > 1.0 to enable 3-phase training:
            SEARCH (fixed high tau) -> HARDEN (tau decreases + entropy penalty) -> SNAP.
            Default tau_search=1.0 disables annealing for backward compatibility.
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
                tree, soft_mse, snapped, snap_mse = EMLTree._train_one(
                    x, y, depth, n_vars, lr, gc, epochs, seed, init, constants,
                    tau_search=tau_search, tau_hard=tau_hard,
                    hardening_fraction=hardening_fraction,
                    lam_entropy=lam_entropy)

                # Choose the better of soft vs snapped
                if snapped is not None and snap_mse < soft_mse:
                    mse, candidate = snap_mse, snapped
                else:
                    mse, candidate = soft_mse, tree

                if mse < depth_best_mse:
                    depth_best_mse, depth_best_tree = mse, candidate
                    stale_seeds = 0
                else:
                    stale_seeds += 1

                if mse < best_mse:
                    best_mse, best_tree = mse, candidate
                    if verbose:
                        w = " (warm)" if warm else ""
                        s = " [snapped]" if candidate is snapped else ""
                        print(f"  depth={depth} seed={seed}{w}{s}: MSE={mse:.6f}")

                if stale_seeds >= 5 or best_mse < 1e-6:
                    break

            prev_best = depth_best_tree
            if best_mse < 1e-6:
                break

        if verbose:
            print(f"Best: depth={best_tree.depth}, MSE={best_mse:.6f}")
        return best_tree
