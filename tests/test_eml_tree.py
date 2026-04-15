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


# ------------------------------------------------------------------
# Tau annealing tests
# ------------------------------------------------------------------

def test_forward_tau_default_unchanged():
    """forward(x) and forward(x, tau=1.0) produce identical results."""
    tree = EMLTree(depth=2, n_vars=1)
    x = torch.randn(50, 1)
    y1 = tree(x)
    y2 = tree(x, tau=1.0)
    assert torch.allclose(y1, y2, atol=1e-7)


def test_forward_tau_sharpens():
    """Low tau reduces leaf entropy (sharper distributions)."""
    tree = EMLTree(depth=2, n_vars=1)
    ent_normal = tree.leaf_entropy(tau=1.0).mean()
    ent_sharp = tree.leaf_entropy(tau=0.1).mean()
    assert ent_sharp < ent_normal


def test_forward_tau_high_uniform():
    """High tau pushes leaf entropy toward maximum."""
    tree = EMLTree(depth=2, n_vars=1)
    max_ent = torch.log(torch.tensor(float(tree.num_candidates)))
    ent_high = tree.leaf_entropy(tau=100.0).mean()
    assert ent_high > max_ent * 0.95


def test_hard_project_produces_one_hot():
    """_hard_project snaps every leaf to a single candidate."""
    tree = EMLTree(depth=2, n_vars=1)
    snapped = EMLTree._hard_project(tree)
    probs = F.softmax(snapped.leaf_logits, dim=-1)
    assert torch.allclose(probs.max(dim=-1).values,
                          torch.ones(tree.num_leaves), atol=1e-5)


def test_hard_project_preserves_argmax():
    """_hard_project preserves which candidate each leaf prefers."""
    tree = EMLTree(depth=2, n_vars=1)
    orig_idx = torch.argmax(tree.leaf_logits, dim=-1)
    snapped = EMLTree._hard_project(tree)
    snap_idx = torch.argmax(snapped.leaf_logits, dim=-1)
    assert torch.equal(orig_idx, snap_idx)


def test_train_one_returns_four_values():
    """_train_one returns (tree, soft_mse, snapped_or_none, snap_mse)."""
    x = torch.linspace(0, 1, 50).unsqueeze(1)
    y = x.squeeze(1) ** 2
    result = EMLTree._train_one(x, y, 2, 1, 0.01, 1.0, 100, seed=0)
    assert len(result) == 4
    tree, soft_mse, snapped, snap_mse = result
    assert snapped is None  # tau_search=1.0 -> no annealing
    assert snap_mse == float("inf")


def test_train_one_with_tau_returns_snapped():
    """_train_one with tau_search > 1 returns a snapped tree."""
    x = torch.linspace(-2, 2, 100).unsqueeze(1)
    y = torch.exp(x.squeeze(1))
    tree, soft_mse, snapped, snap_mse = EMLTree._train_one(
        x, y, 1, 1, 0.01, 1.0, 500, seed=0,
        tau_search=2.5, tau_hard=0.01)
    assert snapped is not None
    assert snap_mse < float("inf")


@pytest.mark.slow
def test_tau_annealing_reduces_entropy():
    """Training with tau annealing produces lower leaf entropy than without."""
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.exp(x.squeeze(1))

    tree_no_tau, _, _, _ = EMLTree._train_one(
        x, y, 1, 1, 0.01, 1.0, 3000, seed=42)
    tree_tau, _, _, _ = EMLTree._train_one(
        x, y, 1, 1, 0.01, 1.0, 3000, seed=42,
        tau_search=2.5, tau_hard=0.01)

    ent_no_tau = tree_no_tau.leaf_entropy().mean().item()
    ent_tau = tree_tau.leaf_entropy().mean().item()
    assert ent_tau < ent_no_tau


@pytest.mark.slow
def test_tau_annealing_converges_exp():
    """fit() with tau annealing still converges for exp(x)."""
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.exp(x.squeeze(1))
    tree = EMLTree.fit(x, y, max_depth=2, n_restarts=5, epochs=5000,
                       tau_search=2.5, tau_hard=0.01)
    assert F.mse_loss(tree(x), y).item() < 1e-2
