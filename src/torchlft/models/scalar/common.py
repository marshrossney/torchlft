from enum import StrEnum, auto
from typing import TypeAlias
from dataclasses import dataclass

from jsonargparse.typing import (
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
    restricted_number_type,
)
import torch

from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.spline import spline_transform
from torchlft.utils.lattice import laplacian
from torchlft.lattice.scalar.action import Phi4Action
from torchlft.lattice.scalar.layers import (
    GlobalRescalingLayer,
    TriangularLinearLayer
)
from torchlft.utils.lattice import checkerboard_mask

Tensor: TypeAlias = torch.Tensor

AtLeastMinusFour = restricted_number_type(
    "AtLeastMinusFour", float, [(">=", -4)]
)


# TODO shouldn't require single lattice length
class ValidPartitioning(StrEnum):
    lexicographic = auto()
    checkerboard = auto()
    random = auto()

    def build(self, lattice_length: int, lattice_dim: int):
        L, d = lattice_length, lattice_dim
        D = pow(L, d)

        if str(self) == "lexicographic":
            return torch.arange(D)

        elif str(self) == "checkerboard":
            checker = checkerboard_mask([L for _ in range(d)]).flatten()
            output_indices = torch.cat(
                [torch.argwhere(checker), torch.argwhere(~checker)]
            ).squeeze(1)
            _, input_indices = output_indices.sort()
            return input_indices

        elif str(self) == "random":
            return torch.randperm(D)


@dataclass
class Phi4Target:
    lattice_length: PositiveInt
    λ: NonNegativeFloat
    β: PositiveFloat | None = None
    m_sq: AtLeastMinusFour | None = None

    def __post_init__(self):
        assert (
            self.lattice_length % 2 == 0
        ), "lattice length must be divisible by 2"
        assert (self.β is None) ^ (
            self.m_sq is None
        ), "Cannot provide both β and m_sq"

    def build(self) -> Phi4Action:
        params = {
            "lattice": (self.lattice_length, self.lattice_length),
            "λ": self.λ,
        }
        if self.β is not None:
            params["β"] = self.β
        else:
            params["m_sq"] = self.m_sq
        return Phi4Action(**params)


@dataclass
class AffineTransform:
    symmetric: bool

    def build(self):
        return affine_transform(symmetric=self.symmetric)


@dataclass
class SplineTransform:
    n_bins: PositiveInt
    bounds: PositiveFloat = 5.0

    def build(self):
        return spline_transform(
            n_bins=self.n_bins,
            lower_bound=-self.bounds,
            upper_bound=+self.bounds,
            boundary_conditions="linear",
        )


@dataclass
class FreeTheoryLayer:
    m_sq: PositiveFloat | None = None
    frozen: bool = True

    def build(self, target: Phi4Target):
        L = target.lattice_length
        kernel = -laplacian(L, 2) + self.m_sq * torch.eye(L**2)
        layer = TriangularLinearLayer.from_gaussian_target(precision=kernel)
        layer.requires_grad_(not self.frozen)
        
        def forward_pre(mod, inputs):
            (inputs,) = inputs
            return inputs.flatten(1)

        def forward_post(mod, inputs, outputs):
            output, ldj = outputs
            return (output.unflatten(1, (L, L, 1)), ldj)

        layer.register_forward_pre_hook(forward_pre)
        layer.register_forward_hook(forward_post)

        return layer


@dataclass
class GlobalRescaling:
    init_scale: PositiveFloat = 1
    frozen: bool = False

    def build(self, target: Phi4Target):
        layer = GlobalRescalingLayer(self.init_scale)
        layer.requires_grad_(not self.frozen)
        return layer
