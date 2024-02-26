import itertools
import math
from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor


def make_2d_striped_mask(
    lattice: tuple[int, int], dim: int, period: int = 2, offset: int = 0
) -> BoolTensor:
    assert dim in (1, 2)  # Dimensions2d
    assert period >= 2

    μ, ν = 1, 2  # Dimensions2d

    k = 1 if dim == ν else 2

    # NOTE: did I put the batch dim in for any reason?
    mask = (
        torch.zeros((1, *lattice), dtype=torch.bool)
        .unflatten(k, (-1, period))
        .index_fill(k + 1, torch.tensor(offset % period), True)
        .flatten(k, k + 1)
    )

    return mask.squeeze(0)


def checkerboard_mask(
    lattice: tuple[int, int],
    offset: int = 0,
    device: torch.device | None = None,
):
    # only works for even-valued lattice lengths
    assert lattice[0] % 2 == 0
    assert lattice[1] % 2 == 0

    checker = torch.zeros(lattice, dtype=torch.bool, device=device)
    checker[::2, ::2] = checker[1::2, 1::2] = True

    if offset % 2 == 1:
        checker = ~checker

    return checker


def dilated_checkerboard_mask(
    lattice: tuple[int, int],
    dilation: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
):
    # only works for even-valued lattice lengths
    assert lattice[0] % 2 == 0
    assert lattice[1] % 2 == 0

    n, m = dilation
    δμ, δν = offset

    checker = torch.zeros(lattice, dtype=torch.bool)
    checker[:: (2 * n), :: (2 * m)] = True  # noqa: E203
    checker[n :: (2 * n), m :: (2 * m)] = True  # noqa: E203

    checker = checker.roll((-δμ, -δν), (0, 1))

    return checker


def laplacian(lattice_length: int, lattice_dim: int) -> Tensor:
    """
    Creates a Laplacian matrix.

    This works by taking the kronecker product of the one-dimensional
    Laplacian matrix with the identity.

    Assumes a square, periodic lattice.
    """
    assert lattice_dim in (1, 2)

    identity = torch.eye(lattice_length)
    lap_1d = (
        -2 * identity  # main diagonal
        + torch.diag(torch.ones(lattice_length - 1), diagonal=1)  # upper
        + torch.diag(torch.ones(lattice_length - 1), diagonal=-1)  # lower
    )
    # periodicity
    lap_1d[0, -1] += 1
    lap_1d[-1, 0] += 1

    if lattice_dim == 1:
        return lap_1d

    lap_2d = torch.kron(lap_1d, identity) + torch.kron(identity, lap_1d)
    return lap_2d


def restore_geometry_2d(matrix: Tensor, lattice: tuple[int, int]) -> Tensor:
    L, T = lattice
    assert matrix.shape[0] == L * T

    return torch.stack(
        [
            row.view(L, T).roll(
                (-(i // T), -(i % L)),
                dims=(0, 1),
            )
            for i, row in enumerate(matrix)
        ],
        dim=0,
    ).mean(dim=0)


def build_neighbour_list(
    lattice: tuple[int, ...],
) -> list[list[int]]:
    indices = torch.arange(math.prod(lattice)).view(lattice)
    dims = range(len(lattice))
    neighbour_indices = torch.stack(
        [
            indices.roll(shift, dim).flatten()
            for shift, dim in itertools.product([1, -1], dims)
        ],
        dim=1,
    )
    return neighbour_indices.tolist()
