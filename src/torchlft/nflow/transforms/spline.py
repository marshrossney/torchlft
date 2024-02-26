from enum import StrEnum
from functools import partial
from math import log
import warnings

import torch
import torch.nn.functional as F

from torchlft.nflow.transforms.wrappers import mask_outside_interval

Tensor = torch.Tensor

warnings.filterwarnings("ignore", message="torch.searchsorted")


def _get_segment(
    input: Tensor, knots: Tensor, inverse: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert input.shape == (1,)
    assert knots.dim() == 1
    assert len(knots) % 3 == 0
    knots_x, knots_y, knots_dydx = knots.tensor_split(3)

    i1 = torch.searchsorted(knots_y if inverse else knots_x, input)

    i0_i1 = torch.cat((i1 - 1, i1)).expand(3, -1)
    knots = torch.stack((knots_x, knots_y, knots_dydx))

    (x0, x1), (y0, y1), (d0, d1) = knots.gather(-1, i0_i1)

    return x0, x1, y0, y1, d0, d1


def _spline_forward(
    x: Tensor,
    params: Tensor,
) -> tuple[Tensor, Tensor]:
    x0, x1, y0, y1, d0, d1 = _get_segment(x, params)
    s = (y1 - y0) / (x1 - x0)

    θx = (x - x0) / (x1 - x0)

    denom = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

    θy = (s * θx**2 + d0 * θx * (1 - θx)) / denom

    y = y0 + (y1 - y0) * θy

    dydx = (
        s**2
        * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
        / denom**2
    )
    return y, dydx


def _spline_inverse(y: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    x0, x1, y0, y1, d0, d1 = _get_segment(y, params, inverse=True)
    s = (y1 - y0) / (x1 - x0)

    θy = (y - y0) / (y1 - y0)

    b = d0 - (d1 + d0 - 2 * s) * θy
    a = s - b
    c = -s * θy

    θx = (-2 * c) / (b + (b**2 - 4 * a * c).sqrt())

    x = x0 + (x1 - x0) * θx

    denom = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

    dydx = (
        s**2
        * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
        / denom**2
    )
    dxdy = 1 / dydx

    return x, dxdy


class SplineBC(StrEnum):
    free = "free"
    linear = "linear"
    periodic = "periodic"


def _generate_knots(
    params: Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
    boundary_conditions: SplineBC,
    min_width: float = 1e-3,
    min_height: float = 1e-3,
    min_deriv: float = 1e-3,
) -> Tensor:
    assert lower_bound < upper_bound
    # assert not (periodic and linear_tails)
    assert params.isfinite().all()
    n_params = params.shape[-1]
    if boundary_conditions == "periodic":
        assert n_params % 3 == 0
        n_bins = n_params // 3
        n_derivs = n_bins
    elif boundary_conditions == "linear":
        assert (n_params + 1) % 3 == 0
        n_bins = (n_params + 1) // 3
        n_derivs = n_bins - 1
    elif boundary_conditions == "free":
        assert (n_params - 1) % 3 == 0
        n_bins = (n_params - 1) // 3
        n_derivs = n_bins + 1
    else:
        raise Exception("invalid bcs")  # TODO custom exception
    assert min_width * n_bins < 1
    assert min_height * n_bins < 1

    widths, heights, derivs = params.split((n_bins, n_bins, n_derivs), dim=-1)

    # Normalise the widths and heights to the interval
    widths = (
        F.softmax(widths, dim=-1) * (1 - min_width * n_bins) + min_width
    ) * (upper_bound - lower_bound)
    heights = (
        F.softmax(heights, dim=-1) * (1 - min_height * n_bins) + min_height
    ) * (upper_bound - lower_bound)

    # Ensure the derivatives are positive and > min_slope
    # Specifying β = log(2) / (1 - ε) means softplus(0, β) = 1
    derivs = F.softplus(derivs, beta=log(2) / (1 - min_deriv)) + min_deriv

    if boundary_conditions == "periodic":
        derivs = torch.cat((derivs, derivs[..., 0:1]), dim=-1)
    elif boundary_conditions == "linear":
        ones = derivs.new_ones(*derivs.shape[:-1], 1)
        derivs = torch.cat((ones, derivs, ones), dim=-1)

    # Build the spline
    zeros = widths.new_zeros((*widths.shape[:-1], 1))
    knots_x = torch.cat(
        (
            zeros,
            torch.cumsum(widths, dim=-1),
        ),
        dim=-1,
    ).add(lower_bound)
    knots_y = torch.cat(
        (
            zeros,
            torch.cumsum(heights, dim=-1),
        ),
        dim=-1,
    ).add(lower_bound)
    knots_dydx = derivs

    knots = torch.cat((knots_x, knots_y, knots_dydx), dim=-1)

    return knots


def spline_transform(
    n_bins: int,
    lower_bound: float,
    upper_bound: float,
    boundary_conditions: SplineBC,
    batch_dims: int = 2,
    min_width: float = 1e-3,
    min_height: float = 1e-3,
    min_deriv: float = 1e-3,
    bounds_tol: float = 1e-4,
):
    generate_knots = partial(
        _generate_knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        boundary_conditions=boundary_conditions,
        min_width=min_width,
        min_height=min_height,
        min_deriv=min_deriv,
    )

    forward_fn = mask_outside_interval(
        _spline_forward,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tol=bounds_tol,
    )

    for dim in range(batch_dims):
        forward_fn = torch.vmap(forward_fn)

    if boundary_conditions == "periodic":
        n_derivs = n_bins
    elif boundary_conditions == "linear":
        n_derivs = n_bins - 1
    elif boundary_conditions == "free":
        n_derivs = n_bins + 1

    class SplineTransform:
        n_params: int = 2 * n_bins + n_derivs

        def __init__(self, params: Tensor):
            self.knots = generate_knots(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.knots)

    return SplineTransform
