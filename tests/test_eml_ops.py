import torch
import math
import pytest
from monolith.eml_ops import safe_eml


def test_eml_exp_identity():
    x = torch.linspace(-5, 5, 100)
    result = safe_eml(x, torch.ones_like(x))
    assert torch.allclose(result, torch.exp(x), atol=1e-6)


def test_eml_zero_left():
    x = torch.linspace(0.1, 5, 100)
    result = safe_eml(torch.zeros_like(x), x)
    assert torch.allclose(result, 1.0 - torch.log(x), atol=1e-6)


def test_eml_euler_constant():
    result = safe_eml(torch.tensor(1.0), torch.tensor(1.0))
    assert abs(result.item() - math.e) < 1e-6


def test_safe_eml_no_nan_large_left():
    assert torch.isfinite(safe_eml(torch.tensor(1000.0), torch.tensor(1.0)))


def test_safe_eml_no_nan_zero_right():
    assert torch.isfinite(safe_eml(torch.tensor(0.0), torch.tensor(0.0)))


def test_safe_eml_no_nan_negative_right():
    assert torch.isfinite(safe_eml(torch.tensor(0.0), torch.tensor(-5.0)))


def test_safe_eml_gradient_flows():
    left = torch.tensor(1.0, requires_grad=True)
    right = torch.tensor(2.0, requires_grad=True)
    safe_eml(left, right).backward()
    assert left.grad is not None and torch.isfinite(left.grad)
    assert right.grad is not None and torch.isfinite(right.grad)
