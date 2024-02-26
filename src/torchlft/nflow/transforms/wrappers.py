from functools import lru_cache
from math import pi as π
from typing import Any, Callable, TypeAlias

import torch

from torchlft.utils.torch import mod_2pi

Tensor: TypeAlias = torch.Tensor
Transform: TypeAlias = Callable[[Tensor, Any, ...], tuple[Tensor, Tensor]]
Wrapper: TypeAlias = Callable[Transform, Transform]
TransformFunc: TypeAlias = Callable  # TODO


def complex_to_arg(transform):
    def complex_to_arg_(x: Tensor, *args, **kwargs):
        r, θ = x.abs(), mod_2pi(x.angle())
        # NOTE: should I demand r == 1 here to be safe??
        ψ, grad_or_ldj = transform(θ, *args, **kwargs)
        y = torch.polar(r, ψ)
        return y, grad_or_ldj

    return complex_to_arg_


def unit_vector_to_angle(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        raise NotImplementedError

    return


def complex_to_vector(transform):
    def complex_to_vector_(x: Tensor, *args, **kwargs):
        assert x.shape[-1] == 1
        x_2d = torch.cat([x.real, x.imag], dim=-1)
        y_2d, grad_or_ldj = transform(x_2d, *args, **kwargs)
        re_y, im_y = y_2d.split(1, dim=-1)
        y = torch.complex(re_y, im_y)
        return y, grad_or_ldj

    return complex_to_vector_


def complex_to_polar(transform):
    def complex_to_polar_(x: Tensor, *args, **kwargs):
        assert x.shape[-1] == 1
        rθ = torch.cat([x.abs(), mod_2pi(x.angle())], dim=-1)
        ρφ, grad_or_ldj = transform(rθ)
        ρ, φ = ρφ.split(1, dim=-1)
        y = torch.polar(ρ, φ)

        # Assume that r -> ρ went first, then θ -> φ
        ldj = grad_or_ldj
        ldj = ldj + ρ.log().flatten(1).sum(1, keepdim=True)

        return y, ldj

    return complex_to_polar_


def vector_to_polar(transform):
    def vector_to_polar_(x: Tensor, *args, **kwargs):
        raise NotImplementedError

    return


def pi_rotation(transform):
    def pi_rotation_(θ: Tensor, *args, **kwargs):
        θ, grad_or_ldj = transform(mod_2pi(θ + π))
        return mod_2pi(θ - π), grad_or_ldj

    return pi_rotation_


def mask_outside_interval(
    transform,
    *,
    lower_bound: float,
    upper_bound: float,
    tol: float,
):
    assert lower_bound < upper_bound

    def mask_outside_interval_(x: Tensor, *args, **kwargs):
        inside_bounds = (x > lower_bound + tol) & (x < upper_bound - tol)

        y, grad = transform(
            x.clamp(lower_bound + tol, upper_bound - tol), *args, **kwargs
        )

        y = torch.where(inside_bounds, y, x)
        # NOTE: require grad, not ldj!
        grad = torch.where(inside_bounds, grad, torch.ones_like(grad))

        return y, grad

    return mask_outside_interval_


def sum_log_gradient(transform):
    def sum_log_gradient_(x: Tensor, *args, **kwargs):
        y, grad = transform(x, *args, **kwargs)
        sum_log_grad = grad.log().flatten(start_dim=1).sum(dim=1, keepdim=True)
        return y, sum_log_grad

    return sum_log_gradient_


@lru_cache
def mixture(
    transform: TransformFunc,
    weighted: bool = True,
    mixture_dim: int = -2,
) -> TransformFunc:
    d = mixture_dim
    vmapped_transform = torch.vmap(transform, (None, d), (d, d))

    if weighted:

        def mixture_(
            x: Tensor, params: Tensor, **kwargs
        ) -> tuple[Tensor, Tensor]:
            params, weights = params.tensor_split([-1], dim=-1)
            # NOTE: vmap/allclose support github.com/pytorch/functorch/issues/275
            # assert torch.allclose(weights.sum(dim=d), torch.ones(1))
            y, dydx = vmapped_transform(x, params, **kwargs)
            assert y.shape == weights.shape
            return (weights * y).sum(dim=d), (weights * dydx).sum(dim=d)

    else:

        def mixture_(
            x: Tensor, params: Tensor, **kwargs
        ) -> tuple[Tensor, Tensor]:
            y, dydx = vmapped_transform(x, params, **kwargs)
            return y.mean(dim=d), dydx.mean(dim=d)

    return mixture_


def mix_with_identity(transform: TransformFunc) -> TransformFunc:
    def mix_with_identity_(
        x: Tensor, params: Tensor, **kwargs
    ) -> tuple[Tensor, Tensor]:
        params, c = params.tensor_split([-1], dim=-1)
        # assert torch.all((c >= 0) and (c <= 1))

        y, dydx = transform(x, params, **kwargs)

        y = c * y + (1 - c) * x
        dydx = c * dydx + (1 - c)

        return y, dydx

    return mix_with_identity_


def dilute(transform, factor: float):
    assert (factor >= 0) and (factor <= 1)
    c = 1 - factor

    def dilute_(x: Tensor, *args, **kwargs):
        y, dydx = transform(x, *args, **kwargs)

        y = c * y + (1 - c) * x
        dydx = c * dydx + (1 - c)

        return y, dydx

    return dilute_


def rescale_to_interval(
    transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
) -> TransformFunc:
    a, b = lower_bound, upper_bound
    assert a < b

    def rescale_to_interval_(x: Tensor, *args, **kwargs):
        y, dydx = transform((x - a) / (b - a), *args, **kwargs)
        return y * (b - a) + a, dydx

    return rescale_to_interval_
