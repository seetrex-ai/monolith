import torch
from torch import Tensor

EXP_CLAMP = (-20.0, 20.0)
LN_CLAMP_MIN = 1e-7


def safe_eml(left: Tensor, right: Tensor) -> Tensor:
    """eml(x, y) = exp(x) - ln(y) with clamping for numerical stability."""
    return torch.exp(left.clamp(*EXP_CLAMP)) - torch.log(right.clamp(min=LN_CLAMP_MIN))
