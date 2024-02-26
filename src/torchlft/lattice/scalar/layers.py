from math import log
from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.nflow.layer import Layer
from torchlft.nflow.nn import permuted_conv2d
from torchlft.nflow.partition import Checkerboard2d
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.utils.linalg import mv
from torchlft.utils.lattice import checkerboard_mask
from torchlft.utils.torch import softplus, inv_softplus


Tensor: TypeAlias = torch.Tensor


class GlobalRescalingLayer(Layer):
    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        init_scale = inv_softplus(torch.tensor(init_scale))
        self.register_parameter("scale", nn.Parameter(init_scale))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        σ = softplus(self.scale) 
        φ = σ * z
        log_det_dφdz = σ.log().mul(z[0].numel()).expand(φ.shape[0], 1)
        return φ, log_det_dφdz


class DiagonalLinearLayer(Layer):
    def __init__(self, init_diag: Tensor):
        super().__init__()
        
        weight = inv_softplus(init_diag)
        self.register_parameter("weight", nn.Parameter(weight))
        
    @classmethod
    def from_size(cls, size: int):
        return cls(torch.eye(size))

    def get_weight(self) -> Tensor:
        return torch.diag_embed(softplus(self.weight))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        assert z.dim() == 2
        D = self.get_weight()
        φ = mv(D, z)
        log_det_dφdz = D.diag().log().sum().expand(φ.shape[0], 1)
        return φ, log_det_dφdz


class TriangularLinearLayer(Layer):
    def __init__(self, init_weights: Tensor):
        super().__init__()
        assert init_weights.dim() == 2
        assert init_weights.shape[0] == init_weights.shape[1]
        assert torch.allclose(init_weights.tril(), init_weights)

        mask = torch.ones_like(init_weights, dtype=torch.bool).tril(-1)
        diag = inv_softplus(init_weights.diag())
        tril = init_weights.masked_select(mask)

        self.register_parameter("diag", nn.Parameter(diag))
        self.register_parameter("tril", nn.Parameter(tril))
        self.register_buffer("mask", mask)

    @classmethod
    def from_size(cls, size: int):
        return cls(torch.eye(D) + torch.empty(D**2).uniform_(0, 1).tril(-1))

    @classmethod
    def from_gaussian_target(cls, covariance: Tensor | None = None, precision: Tensor | None = None):
        assert (covariance is None) ^ (precision is None)
        if precision is not None:
            covariance = torch.linalg.inv(precision)

        cholesky = torch.linalg.cholesky(covariance)

        return cls(cholesky)

    def get_weight(self) -> Tensor:
        return torch.diag_embed(softplus(self.diag)).masked_scatter(self.mask, self.tril)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        assert z.dim() == 2
        L = self.get_weight()
        φ = mv(L, z)
        log_det_dφdz = L.diag().log().sum().expand(φ.shape[0], 1)
        return φ, log_det_dφdz


class DenseCouplingLayer(Layer):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        net: nn.Module,
        layer_id: int,
    ):
        super().__init__()
        self.register_module("net", net)
        self.register_module("transform", transform)

        self.layer_id = layer_id

    def split(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        φ_A, φ_B = φ_in.tensor_split(2, dim=1)
        if self.layer_id % 2:
            return φ_B, φ_A
        else:
            return φ_A, φ_B

    def join(self, φ_a: Tensor, φ_p: Tensor) -> tuple[Tensor, Tensor]:
        if self.layer_id % 2:
            return torch.cat([φ_p, φ_a], dim=1)
        else:
            return torch.cat([φ_a, φ_p], dim=1)

    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        N, D = φ_in.shape
        assert D % 2 == 0

        φ_a, φ_p = self.split(φ_in)

        # Construct conditional transformation
        context = self.net(φ_p).view(*φ_a.shape, -1)
        transform = self.transform(context)

        # Transform active variables
        φ_a, ldj = transform(φ_a.unsqueeze(-1))

        φ_out = self.join(φ_a.squeeze(-1), φ_p)

        return φ_out, ldj


class ConvCouplingLayer(Layer):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        spatial_net: nn.Module,
        layer_id: int,
    ):
        super().__init__()

        partitioning = Checkerboard2d(partition_id=layer_id)
        self.register_module("partitioning", partitioning)

        self.register_module("spatial_net", spatial_net)
        self.register_module("transform", transform)

    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        # Get active and frozen masks
        _, L, T, _ = φ_in.shape
        active_mask, frozen_mask = self.partitioning(dimensions=(L, T))

        # Construct conditional transformation
        net_inputs = frozen_mask.unsqueeze(-1) * φ_in
        context = self.spatial_net(net_inputs)[:, active_mask]
        transform = self.transform(context)

        # Transform active variables
        φ_out = φ_in.clone()
        φ_out[:, active_mask], ldj = transform(φ_in[:, active_mask])

        return φ_out, ldj


class CouplingLayer(Layer):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        radius: int,
        layer_id: int,
    ):
        super().__init__()

        partitioning = Checkerboard2d(partition_id=layer_id)
        self.register_module("partitioning", partitioning)

        self.register_module("transform", transform)

        # With shifts
        K = 2 * radius + 1
        shifts = torch.stack(
            torch.meshgrid(
                torch.arange(-radius, radius + 1),
                torch.arange(-radius, radius + 1),
                indexing="xy",
            ),
            dim=-1,
        )
        checker = checkerboard_mask([K + 1, K + 1], offset=1)[:-1, :-1]
        shifts = shifts[checker]
        self.shifts = shifts.tolist()

        # With conv
        elements = torch.zeros(K, K, dtype=torch.long).masked_scatter(
            checker, torch.arange(1, K**2 // 2 + 1)
        )
        kernels = nn.functional.one_hot(elements)[..., 1:]
        self.register_buffer("kernels", kernels.float())

    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        # Get active and frozen masks
        _, L, T, _ = φ_in.shape
        active_mask, frozen_mask = self.partitioning(dimensions=(L, T))

        # Mask active elements when constructing context
        context = frozen_mask.unsqueeze(-1) * φ_in

        # Stack context within receptive field
        context = torch.cat(
            [context.roll(shift, (1, 2)) for shift in self.shifts],
            dim=-1,
        )

        # NOTE: Alternative method, seems slightly slower
        # kernels = self.kernels.unsqueeze(-1).expand(-1, -1, -1, context.shape[-1])
        # context = permuted_conv2d(context, kernels)

        # Select active part
        context = context[:, active_mask]
        transform = self.transform(context)

        # Transform active variables
        φ_out = φ_in.clone()
        φ_out[:, active_mask], ldj = transform(φ_in[:, active_mask])

        return φ_out, ldj


class UpscalingLayer(Layer):
    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        _, L, T, n = φ_in.shape
        assert n % 4 == 0

        # TODO: see https://github.com/marshrossney/multilevel-flow-experiments/blob/basic-multilevel/notebooks/basic_multilevel.ipynb
