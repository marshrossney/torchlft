from functools import partial
from math import log, pi as π
from typing import Callable, NamedTuple, Self, TypeAlias

import torch
import torch.nn as nn

from torchlft.lattice.action import TargetAction
from torchlft.utils.lattice import laplacian
from torchlft.utils.linalg import dot, mv

Tensor: TypeAlias = torch.Tensor


class GaussianAction(TargetAction):
    def __init__(self, lattice_length: int, lattice_dim: int, m_sq: float):

        assert lattice_length % 2 == 0
        assert lattice_dim in (1, 2)
        assert m_sq > 0

        L, d = lattice_length, lattice_dim

        super().__init__(lattice=tuple(L for _ in range(d)), m_sq=m_sq)

        D = pow(L, d)
        K = -laplacian(L, d) + m_sq * torch.eye(D)
        Σ = torch.linalg.inv(K)
        C = torch.linalg.cholesky(Σ)

        log_abs_det_C = C.diag().log().sum()

        # check |Σ| = |C|^2
        assert torch.allclose(
            2 * log_abs_det_C,
            torch.linalg.slogdet(Σ)[1],
            atol=1e-5,
        )

        self.lattice_length = L
        self.lattice_dim = d
        self.lattice_size = D
        self.m_sq = m_sq
        self.log_norm = 0.5 * D * log(2 * π) + log_abs_det_C

        self.register_buffer("kernel", K)
        self.register_buffer("covariance", Σ)
        self.register_buffer("cholesky", C)

    def forward(self, φ: Tensor) -> Tensor:
        K = self.kernel
        S = 0.5 * dot(φ, mv(K, φ))
        return S + self.log_norm

    def grad(self, φ: Tensor) -> Tensor:
        K = self.kernel
        return mv(K, φ)


# TODO check valid ?
class Phi4Couplings(NamedTuple):
    r"""
    Coefficients for the three terms in the Phi^4 action.
    """

    β: float
    α: float
    λ: float

    @classmethod
    def particle_phys(cls, m_sq: float, λ: float) -> Self:
        """
        Standard Particle Physics parametrisation of Phi^4.
        """
        return cls(β=1, α=(4 + m_sq) / 2, λ=λ)

    @classmethod
    def ising(cls, β: float, λ: float) -> Self:
        """
        Ising-like parametrisation of Phi^4.
        """
        return cls(β=β, α=1 - 2 * λ, λ=λ)


def _parse_couplings(couplings: dict[str, float]) -> Phi4Couplings:
    if "β" in couplings:
        return Phi4Couplings.ising(**couplings)
    else:
        return Phi4Couplings.particle_phys(**couplings)


class Phi4Action(TargetAction):

    @property
    def _parsed_couplings(self) -> Phi4Couplings:
        return _parse_couplings(self.couplings)

    def forward(self, φ: Tensor) -> Tensor:
        β, α, λ = self._parsed_couplings
        lattice_dims = (1, 2)

        s = torch.zeros_like(φ)

        # Nearest neighbour interaction
        for μ in lattice_dims:
            s -= β * (φ * φ.roll(-1, μ))

        # phi^2 term
        φ_sq = φ**2
        s += α * φ_sq

        # phi^4 term
        s += λ * φ_sq**2

        # Sum over lattice sites
        return s.sum(dim=lattice_dims)

    def grad(self, φ: Tensor) -> Tensor:
        β, α, λ = self._parsed_couplings
        lattice_dims = (1, 2)

        dsdφ = torch.zeros_like(φ)

        # Nearest neighbour interaction: +ve and -ve shifts
        for μ in lattice_dims:
            dsdφ -= β * (φ.roll(-1, μ) + φ.roll(+1, μ))

        dsdφ += 2 * α * φ

        dsdφ += 4 * λ * φ**3

        return dsdφ


def _local_action(
    φ: float,
    neighbours: list[float],
    couplings: Phi4Couplings,
) -> float:
    """
    Computes the local Phi^4 action for a single-component scalar field.
    """
    φi, φj, β, α, λ = φ, neighbours, *couplings
    return (-β / 2) * φi * sum(φj) + α * φi**2 + λ * φi**4


def get_local_action(
    couplings: dict[str, float]
) -> Callable[[float, list[float]], float]:
    return partial(_local_action, couplings=_parse_couplings(couplings))
