import torch
import sympy
import pytest
from monolith.symbolic import _leaf_to_expr, _build_eml_tree


def test_leaf_to_expr_snaps_discrete():
    x = sympy.Symbol("x")
    expr = _leaf_to_expr(torch.tensor([0.0, 0.0, 10.0]), [x])
    assert expr == x


def test_leaf_to_expr_snaps_to_one():
    x = sympy.Symbol("x")
    assert _leaf_to_expr(torch.tensor([0.0, 10.0, 0.0]), [x]) == sympy.Integer(1)


def test_leaf_to_expr_snaps_to_zero():
    x = sympy.Symbol("x")
    assert _leaf_to_expr(torch.tensor([10.0, 0.0, 0.0]), [x]) == sympy.Integer(0)


def test_leaf_to_expr_mixed():
    x = sympy.Symbol("x")
    expr = _leaf_to_expr(torch.tensor([0.0, 0.5, 0.5]), [x])
    assert isinstance(expr, sympy.Basic) and x in expr.free_symbols


def test_build_eml_tree_depth1():
    x = sympy.Symbol("x")
    result = _build_eml_tree([x, sympy.Integer(1)], depth=1)
    assert sympy.simplify(result) == sympy.exp(x)


def test_build_eml_tree_depth2():
    import math
    x = sympy.Symbol("x")
    leaves = [x, sympy.Integer(1), sympy.Integer(0), sympy.Integer(1)]
    result = _build_eml_tree(leaves, depth=2)
    val = float(result.subs(x, 0.5))
    expected = math.exp(math.exp(0.5)) - math.log(1)
    assert abs(val - expected) < 1e-6


@pytest.mark.slow
def test_to_symbolic_exp():
    from monolith import EMLTree
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.exp(x.squeeze(1))
    tree = EMLTree.fit(x, y, max_depth=1, n_restarts=5, epochs=3000)
    assert sympy.simplify(tree.to_symbolic()) == sympy.exp(sympy.Symbol("x"))


@pytest.mark.slow
def test_to_symbolic_returns_expr():
    from monolith import EMLTree
    torch.manual_seed(42)
    x = torch.linspace(0.5, 3.0, 200).unsqueeze(1)
    y = x.squeeze(1) ** 2
    tree = EMLTree.fit(x, y, max_depth=2, n_restarts=5, epochs=5000)
    expr = tree.to_symbolic()
    assert isinstance(expr, sympy.Basic) and len(expr.free_symbols) > 0


@pytest.mark.slow
def test_to_symbolic_evaluates_close():
    import numpy as np
    from monolith import EMLTree
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.exp(x.squeeze(1))
    tree = EMLTree.fit(x, y, max_depth=1, n_restarts=3, epochs=3000)
    f = sympy.lambdify(sympy.Symbol("x"), tree.to_symbolic(), modules=["numpy"])
    mse = float(np.mean((f(np.linspace(-2, 2, 200)) - tree(x).detach().numpy()) ** 2))
    assert mse < 1e-2


@pytest.mark.slow
def test_snap_symbolic_exp():
    from monolith import EMLTree
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.exp(x.squeeze(1))
    tree = EMLTree.fit(x, y, max_depth=1, n_restarts=5, epochs=3000)
    expr = tree.snap_symbolic(x, y)
    assert expr is not None
    assert sympy.simplify(expr) == sympy.exp(sympy.Symbol("x"))


@pytest.mark.slow
def test_snap_symbolic_fails_gracefully():
    from monolith import EMLTree
    torch.manual_seed(42)
    x = torch.linspace(0.5, 3.0, 200).unsqueeze(1)
    y = x.squeeze(1) ** 2
    tree = EMLTree.fit(x, y, max_depth=2, n_restarts=5, epochs=5000)
    assert tree.snap_symbolic(x, y, tol=2.0) is None
