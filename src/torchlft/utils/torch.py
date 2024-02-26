from collections.abc import Iterable
from math import log, pi as π

import torch

Tensor = torch.Tensor

log2 = log(2)

def mod_2pi(θ: Tensor) -> Tensor:
    return torch.remainder(θ, 2 * π)


def log_cosh(x: Tensor) -> Tensor:
    """Numerically stable implementation of log(cosh(x))"""
    return abs(x) + torch.log1p(torch.exp(-1 * abs(x))) - log(2)


def softplus(x: Tensor, β: float = log2) -> Tensor:
    return (1 / β) * torch.log1p(torch.exp(β * x))

def inv_softplus(σ: Tensor, β: float = log2) -> Tensor:
    """Numerical stable implementation of log( exp(βx) - 1 ) / β"""
    return σ + (1 / β) * torch.log(-torch.expm1(-β * σ))

def sum_except_batch(x: Tensor, keepdim: bool = False) -> Tensor:
    return x.flatten(start_dim=1).sum(dim=1, keepdim=keepdim)


def _as_real(z: Tensor) -> Tensor:
    return torch.view_as_real(z).flatten(start_dim=-2)


def as_real(
    inputs: Tensor | tuple[Tensor, ...]
) -> Tensor | tuple[Tensor, ...]:
    if isinstance(inputs, Tensor):
        return _as_real(inputs)
    return (_as_real(t) for t in inputs)


def tuple_clone(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return type(tensors)([t.clone() for t in tensors])


def tuple_concat(
    tuples: Iterable[tuple[Tensor, ...]], dim: int = 0
) -> tuple[Tensor, ...]:
    """
    Concatenation of tuples of tensors.

    Example:

        >>> a, b, c = (
                (torch.rand(1), torch.rand(10), torch.rand(1, 10))
                for _ in range(3)
        )
        >>> x, y, z = tuple_concat([a, b, c])
        >>> x.shape, y.shape, z.shape
        (torch.Size([3]), torch.Size([30]), torch.Size([3, 10]))

    """
    # return type(tuples)(torch.cat(tensors, dim=dim) for tensors in map(list, zip(*tuples)))
    return type(tuples)(
        torch.cat(tensors, dim=dim) for tensors in zip(*tuples)
    )


def _tuple_stack(
    tensors: list[tuple[Tensor, ...]], dim: int = 0
) -> tuple[Tensor, ...]:
    return tuple([torch.stack(t, dim=dim) for t in zip(*tensors)])


def tuple_stack(tensors: tuple[tuple[Tensor, ...]], dim: int = 0) -> tuple:
    return type(tensors)(
        [
            (
                torch.stack(t, dim=dim)
                if isinstance(t[0], torch.Tensor)
                else tuple_stack(t, dim=dim)
            )
            for t in zip(*tensors)
        ]
    )


def dict_concat(
    dicts: Iterable[dict[str, Tensor]], dim: int = 0, strict: bool = True
) -> dict[str, Tensor]:
    """
    Dim-zero concatenation of dicts of torch.Tensors.

    Examples:

        Dicts with matching keys:

        >>> a, b, c = (
                {"x": torch.rand(1), "y": torch.rand(10), z: torch.rand(1, 10)}
                for _ in range(3)
        )
        >>> out = dict_concat(a, b, c)
        >>> out["x"].shape, out["y"].shape, out["z"].shape
        (torch.Size([3]), torch.Size([30]), torch.Size([3, 10]))

        Dicts without matching keys:

        >>> d = {"x": torch.rand(1), "w": torch.rand(100)}
        >>> dict_concat(a, b, c, d)
        KeyError: 'y'
        >>> out = dict_concat(a, b, c, d, strict=False)
        >>> out["x"].shape, out["y"].shape, out["z"].shape, out["w"].shape
        (torch.Size([4]), torch.Size([30]), torch.Size([3, 10]), torch.Size([100]))  # noqa: E501
    """
    keys = set([k for d in dicts for k in d.keys()])
    if strict:
        return {k: torch.cat([d[k] for d in dicts], dim=dim) for k in keys}
    else:
        return {
            k: torch.cat([d.get(k, torch.tensor([]), dim=dim) for d in dicts])
            for k in keys
        }


def dict_stack(
    dicts: Iterable[dict[str, Tensor]], dim: int = 0, strict: bool = True
) -> dict[str, Tensor]:
    keys = set([k for d in dicts for k in d.keys()])
    if strict:
        return {k: torch.stack([d[k] for d in dicts], dim=dim) for k in keys}
    else:
        return {
            k: torch.stack(
                [d.get(k, torch.tensor([]), dim=dim) for d in dicts]
            )
            for k in keys
        }
