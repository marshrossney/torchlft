from math import log

import torch

from torchlft.nflow.transforms.core import Transform

Tensor = torch.Tensor


def _affine_forward(x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    s, t = params.split(1, dim=-1)
    return x * s + t, s


def _affine_inverse(y: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    s, t = params.split(1, dim=-1)
    s_inv = 1 / s
    return (y - t) * s_inv, s_inv


def affine_transform(
    scale_fn: str = "exponential",
    symmetric: bool = False,
    shift_only: bool = False,
    rescale_only: bool = False,
):
    assert not (shift_only and rescale_only)

    if shift_only:
        n_params = 1

        def handle_params(params: Tensor) -> Tensor:
            return torch.cat([torch.ones_like(params), params], dim=-1)

    else:
        if scale_fn == "exponential":
            scale_fn = lambda s: torch.exp(-s)  # noqa E731
        elif scale_fn == "softplus":
            scale_fn = lambda s: torch.nn.functional.softplus(
                -s, beta=log(2)
            )  # noqa E731

        if symmetric:
            _scale_fn = scale_fn
            scale_fn = lambda s: _scale_fn(s.abs())
            # scale_fn = _scale_fn(s) + _scale_fn(-s))  # noqa E731

        if rescale_only:
            n_params = 1

            def handle_params(params: Tensor) -> Tensor:
                return torch.cat(
                    [scale_fn(params), torch.zeros_like(params)], dim=-1
                )

        else:
            n_params = 2

            def handle_params(params: Tensor) -> Tensor:
                s, t = params.split(1, dim=-1)
                return torch.cat([scale_fn(s), t], dim=-1)

    _n_params = n_params

    class AffineTransform(Transform):
        n_params = _n_params
        handle_params_fn = staticmethod(handle_params)
        transform_fn = staticmethod(_affine_forward)

    return AffineTransform
