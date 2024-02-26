from math import prod
from typing import TypeAlias
from dataclasses import dataclass

from jsonargparse.typing import PositiveInt
import torch
import torch.nn as nn

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.layer import Composition
from torchlft.nflow.nn import DenseNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.nflow.utils import compute_grad_pullback
from torchlft.lattice.scalar.layers import DenseCouplingLayer
from torchlft.utils.linalg import dot

from .common import (
    Phi4Target,
    AffineTransform,
    SplineTransform,
    GlobalRescaling,
    FreeTheoryLayer,
    ValidPartitioning,
)

Tensor: TypeAlias = torch.Tensor


@dataclass
class AffineCouplingBlock:
    transform: AffineTransform
    net: DenseNet
    n_layers: PositiveInt

    def build(self, target: Phi4Target):
        layers = []

        for layer_id in range(self.n_layers):

            transform_module = UnivariateTransformModule(
                transform_cls=self.transform.build(),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )

            net = self.net.build()
            size_out = transform_module.transform_cls.n_params * (
                target.lattice_length**2 // 2
            )
            net.append(nn.LazyLinear(size_out, bias=self.net.bias))

            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        return Composition(*layers)


@dataclass
class SplineCouplingBlock:
    transform: SplineTransform
    net: DenseNet
    n_layers: PositiveInt

    def build(self, target: Phi4Target):
        layers = []

        for layer_id in range(self.n_layers):

            transform_module = UnivariateTransformModule(
                transform_cls=self.transform.build(),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )

            net = self.net.build()
            size_out = transform_module.transform_cls.n_params * (
                target.lattice_length**2 // 2
            )
            net.append(nn.LazyLinear(size_out, bias=self.net.bias))

            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        return Composition(*layers)


@dataclass
class AffineSplineFlow:
    to_free: FreeTheoryLayer | None = None
    affine: AffineCouplingBlock | None = None
    spline: SplineCouplingBlock | None = None
    rescale: GlobalRescaling | None = None

    def build(self, target: Phi4Target):
        blocks = []
        for block in (self.to_free, self.affine, self.spline, self.rescale):
            if block is not None:
                blocks.append(block.build(target))
        return Composition(*blocks)


class DenseCouplingModel(BaseModel):
    def __init__(
        self,
        flow: AffineSplineFlow,
        target: Phi4Target,
        partitioning: ValidPartitioning,
    ):
        super().__init__()

        self.register_module("target", target.build())
        self.register_module("flow", flow.build(target))

        # NOTE: currently requires L1 = L2 = L
        partitioning = ValidPartitioning(str(partitioning))
        indices = partitioning.build(target.lattice_length, 2)
        self.register_buffer("indices", indices)

    def sample_base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        D = prod(self.target.lattice)

        z = torch.empty(
            size=(batch_size, D), device=self.device, dtype=self.dtype
        ).normal_()

        S = 0.5 * dot(z, z)

        return z, S.unsqueeze(-1)

    def compute_target(self, φ: Tensor) -> Tensor:
        φ = φ.unflatten(1, (*self.target.lattice, 1))
        return self.target(φ)

    def flow_forward(self, z: Tensor) -> Tensor:
        φ, ldj = self.flow(z)
        φ = φ[:, self.indices]  # interleave elements to undo partitioning
        return φ, ldj

    def grad_pullback(self, z: Tensor) -> Tensor:
        return compute_grad_pullback(self, z)
