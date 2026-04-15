import torch
import sympy
from sympy import exp, log, Integer, Float, Symbol


def _leaf_to_expr(logits, var_symbols, snap_threshold=0.95, constants=(0.0, 1.0)):
    """Convert leaf logits to a SymPy expression, snapping if confidence is high."""
    probs = torch.softmax(logits, dim=0).detach().tolist()

    const_syms = []
    for c in constants:
        if c == int(c):
            const_syms.append(sympy.Rational(c).limit_denominator(1000))
        elif abs(c - 3.141592653589793) < 1e-10:
            const_syms.append(sympy.pi)
        else:
            const_syms.append(Float(round(c, 6)))

    candidates = const_syms + list(var_symbols)
    best = max(probs)

    if best > snap_threshold:
        return candidates[probs.index(best)]

    result = Integer(0)
    for p, c in zip(probs, candidates):
        if p >= 0.001:
            result += Float(round(p, 4)) * c
    return result


def _build_eml_tree(leaf_exprs, depth):
    """Bottom-up EML tree construction, mirroring the forward pass."""
    nodes = list(leaf_exprs)
    for _ in range(depth):
        nodes = [exp(nodes[i]) - log(nodes[i + 1])
                 for i in range(0, len(nodes), 2)]
    return nodes[0]


def to_symbolic(tree):
    """Reconstruct the SymPy expression computed by a trained EMLTree."""
    syms = [Symbol("x")] if tree.n_vars == 1 else \
           [Symbol(f"x_{i+1}") for i in range(tree.n_vars)]

    leaves = [_leaf_to_expr(tree.leaf_logits[i], syms, constants=tree.constants)
              for i in range(tree.num_leaves)]
    return sympy.simplify(_build_eml_tree(leaves, tree.depth))


def snap_symbolic(tree, x_data, y_data, tol=2.0):
    """Force-snap decompilation: argmax all leaves, verify against data."""
    import numpy as np

    syms = [Symbol("x")] if tree.n_vars == 1 else \
           [Symbol(f"x_{i+1}") for i in range(tree.n_vars)]

    leaves = [_leaf_to_expr(tree.leaf_logits[i], syms, snap_threshold=0.0,
                            constants=tree.constants)
              for i in range(tree.num_leaves)]

    expr = sympy.simplify(_build_eml_tree(leaves, tree.depth))

    if x_data.dim() == 1:
        x_data = x_data.unsqueeze(1)

    try:
        fn = sympy.lambdify(list(syms), expr, modules=["numpy"])
        if tree.n_vars == 1:
            y_snap = np.array(fn(x_data.squeeze(1).detach().numpy()), dtype=np.float64)
        else:
            cols = [x_data[:, i].detach().numpy() for i in range(tree.n_vars)]
            y_snap = np.array(fn(*cols), dtype=np.float64)
    except (ValueError, ZeroDivisionError, OverflowError, KeyError, TypeError):
        return None

    if not np.all(np.isfinite(y_snap)):
        return None

    mse_snap = float(np.mean((y_snap - y_data.detach().numpy()) ** 2))
    mse_orig = float(torch.nn.functional.mse_loss(tree(x_data), y_data).item())

    if mse_orig == 0:
        return expr if mse_snap < 1e-6 else None
    return expr if mse_snap / mse_orig < tol else None
