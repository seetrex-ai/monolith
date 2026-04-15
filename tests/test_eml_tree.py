import torch
import torch.nn.functional as F
import pytest
from monolith import EMLTree


def test_parameter_count_depth4_nvars1():
    tree = EMLTree(depth=4, n_vars=1)
    assert sum(p.numel() for p in tree.parameters()) == 48


def test_parameter_count_depth3_nvars3():
    tree = EMLTree(depth=3, n_vars=3)
    assert sum(p.numel() for p in tree.parameters()) == 40


def test_leaf_logits_shape():
    tree = EMLTree(depth=4, n_vars=2)
    assert tree.leaf_logits.shape == (16, 4)


def test_forward_shape_1var():
    tree = EMLTree(depth=3, n_vars=1)
    assert tree(torch.randn(100, 1)).shape == (100,)


def test_forward_shape_3var():
    tree = EMLTree(depth=3, n_vars=3)
    assert tree(torch.randn(100, 3)).shape == (100,)


def test_forward_no_nan():
    tree = EMLTree(depth=4, n_vars=1)
    y = tree(torch.linspace(-5, 5, 200).unsqueeze(1))
    assert torch.all(torch.isfinite(y))


def test_forward_gradient_flow():
    tree = EMLTree(depth=3, n_vars=1)
    tree(torch.randn(50, 1)).sum().backward()
    for name, p in tree.named_parameters():
        assert p.grad is not None and torch.all(torch.isfinite(p.grad)), name


def test_forward_single_sample():
    tree = EMLTree(depth=2, n_vars=1)
    y = tree(torch.tensor([[1.0]]))
    assert y.shape == (1,) and torch.isfinite(y)


@pytest.mark.slow
def test_convergence_exp():
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.exp(x.squeeze(1))
    tree = EMLTree.fit(x, y, max_depth=2, n_restarts=5, epochs=5000)
    assert F.mse_loss(tree(x), y).item() < 1e-3


@pytest.mark.slow
def test_convergence_x_squared():
    x = torch.linspace(0.5, 2.5, 200).unsqueeze(1)
    y = x.squeeze(1) ** 2
    tree = EMLTree.fit(x, y, max_depth=3, n_restarts=20, epochs=10000)
    assert F.mse_loss(tree(x), y).item() < 0.01


@pytest.mark.slow
def test_convergence_ln():
    x = torch.linspace(0.5, 3.0, 200).unsqueeze(1)
    y = torch.log(x.squeeze(1))
    tree = EMLTree.fit(x, y, max_depth=3, n_restarts=20, epochs=10000)
    assert F.mse_loss(tree(x), y).item() < 1e-3


def test_leaf_entropy_shape():
    assert EMLTree(depth=3, n_vars=1).leaf_entropy().shape == (8,)


def test_leaf_entropy_high_at_init():
    tree = EMLTree(depth=3, n_vars=1)
    max_ent = torch.log(torch.tensor(3.0))
    assert torch.all(tree.leaf_entropy() > max_ent * 0.8)


def test_leaf_entropy_low_after_sharp_logits():
    tree = EMLTree(depth=2, n_vars=1)
    with torch.no_grad():
        tree.leaf_logits.fill_(0.0)
        tree.leaf_logits[:, 1] = 10.0
    assert torch.all(tree.leaf_entropy() < 0.1)
