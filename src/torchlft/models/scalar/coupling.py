from math import prod
from typing import TypeAlias
from dataclasses import dataclass

from jsonargparse.typing import PositiveInt
import torch

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.layer import Composition
from torchlft.nflow.nn import PointNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.nflow.utils import compute_grad_pullback
from torchlft.lattice.scalar.layers import CouplingLayer
from torchlft.utils.torch import sum_except_batch

from .common import (
    Phi4Target,
    AffineTransform,
    SplineTransform,
    GlobalRescaling,
    FreeTheoryLayer,
)

Tensor: TypeAlias = torch.Tensor


@dataclass
class AffineCouplingBlock:
    transform: AffineTransform
    point_net: PointNet
    radius: PositiveInt
    n_layers: PositiveInt

    def __post_init__(self):
        transform_cls = self.transform.build()
        self.point_net.channels.append(transform_cls.n_params)

    def build(self, *_, **__):
        layers = []

        for layer_id in range(self.n_layers):

            transform_module = UnivariateTransformModule(
                transform_cls=self.transform.build(),
                context_fn=self.point_net.build(),
                wrappers=[sum_log_gradient],
            )
            layer = CouplingLayer(transform_module, self.radius, layer_id)
            layers.append(layer)

        return Composition(*layers)


@dataclass
class SplineCouplingBlock:
    transform: SplineTransform
    point_net: PointNet
    radius: PositiveInt
    n_layers: PositiveInt

    def __post_init__(self):
        transform_cls = self.transform.build()
        self.point_net.channels.append(transform_cls.n_params)

    def build(self, *_, **__):
        layers = []

        for layer_id in range(self.n_layers):

            transform_module = UnivariateTransformModule(
                transform_cls=self.transform.build(),
                context_fn=self.point_net.build(),
                wrappers=[sum_log_gradient],
            )
            layer = CouplingLayer(transform_module, self.radius, layer_id)
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


class CouplingModel(BaseModel):
    def __init__(
        self,
        flow: AffineSplineFlow,
        target: Phi4Target,
    ):
        super().__init__()

        self.register_module("target", target.build())
        self.register_module("flow", flow.build(target))

    def sample_base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        D = prod(self.target.lattice)

        z = torch.empty(
            size=(batch_size, *self.target.lattice, 1), device=self.device, dtype=self.dtype
        ).normal_()

        S = sum_except_batch((z**2 / 2), keepdim=True)

        return z, S

    def compute_target(self, φ: Tensor) -> Tensor:
        return self.target(φ)

    def flow_forward(self, z: Tensor) -> Tensor:
        φ, ldj = self.flow(z)
        return φ, ldj

    def grad_pullback(self, z: Tensor) -> Tensor:
        return compute_grad_pullback(self, z)
