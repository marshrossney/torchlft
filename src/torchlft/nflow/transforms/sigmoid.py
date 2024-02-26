"""
Credit to https://arxiv.org/pdf/2110.00351.pdf
and https://github.com/noegroup/bgflow for this transformation
"""

from functools import partial
from math import pi as π
from typing import Any, Callable, TypeAlias

import torch
import torch.nn.functional as F

from .wrappers import (
    rescale_to_interval,
    mixture,
    mix_with_identity,
)
from .utils import normalise_weights, normalise_single_weight

Tensor: TypeAlias = torch.Tensor
TransformFunc: TypeAlias = Callable[
    [Tensor, Tensor, Any, ...], tuple[Tensor, Tensor]
]
Transform: TypeAlias = Callable[Tensor, tuple[Tensor, Tensor]]


def exponential_ramp(
    x: Tensor, params: Tensor, *, power: int, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    assert isinstance(power, int) and power > 0
    a, b, ε = params, power, eps
    x_masked = torch.where(x > ε, x, torch.full_like(x, ε))
    exp_factor = -a * x_masked.pow(-b)
    ρ = torch.where(
        x > ε,
        torch.exp(exp_factor) / torch.exp(-a),
        torch.zeros_like(x),
    )
    # NOTE: don't need a `where` since dρdx=0 where x<ε already
    dρdx = a * b * x_masked.pow(-(b + 1)) * ρ
    return ρ, dρdx


def sigmoid(ramp: TransformFunc) -> TransformFunc:
    def sigmoid_(x: Tensor, params: Tensor, **kwargs):
        ρ_x, dρdx_x = ramp(x, params, **kwargs)
        ρ_1mx, dρdx_1mx = ramp(1 - x, params, **kwargs)

        σ = ρ_x / (ρ_x + ρ_1mx)

        dσdx = (ρ_1mx * dρdx_x + ρ_x * dρdx_1mx) / (ρ_x + ρ_1mx) ** 2

        return σ, dσdx

    return sigmoid_


def affine(
    sigmoid: TransformFunc,
) -> TransformFunc:
    def affine_(x: Tensor, params: Tensor, **kwargs):
        params, α, β = params.tensor_split([-2, -1], dim=-1)
        σ, dσdx = sigmoid((x - β) * α + 0.5, params, **kwargs)
        return σ, α * dσdx

    return affine_


def build_sigmoid_transform(
    n_mixture: int,
    weighted: bool = True,
    ramp_pow: int = 2,
    min_weight: float = 1e-2,
) -> Transform:
    weighted = weighted if n_mixture > 1 else False

    ramp = partial(exponential_ramp, power=ramp_pow)

    # NOTE: I seem to remember the identity mixture was redundant!
    transform = mixture(
        rescale_to_interval(
            mix_with_identity(affine(sigmoid(ramp))),
            lower_bound=0.0,
            upper_bound=2 * π,
        ),
        weighted=weighted,
        mixture_dim=-2,
    )

    funcs = [
        lambda x: F.softplus(x) + 1e-3,  # exponential ramp 'a'
        lambda x: x.negative().exp() + 1e-3,  # affine 'α'
        # lambda x: F.softplus(x) + 1e-3,  # affine 'α'
        torch.sigmoid,  # affine 'β'
        partial(
            normalise_single_weight, min=min_weight
        ),  # weight wrt identity transform
    ]
    if weighted:
        funcs.append(partial(normalise_weights, dim=-2, min=min_weight))

    def handle_params(params: Tensor) -> Tensor:
        params = params.unflatten(-1, (n_mixture, -1)).split(1, dim=-1)
        params = torch.cat(
            [func(param) for func, param in zip(funcs, params, strict=True)],
            dim=-1,
        )
        return params

    class SigmoidTransform:
        n_params = (4 + int(weighted)) * n_mixture

        def __init__(self, params: Tensor):
            self.params = handle_params(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return transform(x, self.params)

    return SigmoidTransform
