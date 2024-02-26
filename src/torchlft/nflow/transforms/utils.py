import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def normalise_weights(weights: Tensor, dim: int, min: float = 1e-2) -> Tensor:
    d, ε = dim, min
    n_mix = weights.shape[d]
    assert n_mix * ε < 1
    return F.softmax(weights, dim=d) * (1 - n_mix * ε) + ε


def normalise_single_weight(weight: Tensor, min: float = 1e-2) -> Tensor:
    ε = min
    return torch.sigmoid(weight) * (1 - ε) + ε
