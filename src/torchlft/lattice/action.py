from abc import ABCMeta, abstractmethod
from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.nflow.model import Model

Tensor: TypeAlias = torch.Tensor
Tensors: TypeAlias = tuple[Tensor, ...]


class Action(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs: Tensor | Tensors) -> Tensor: ...

    @abstractmethod
    def grad(self, inputs: Tensor | Tensors) -> Tensor | Tensors: ...


class TargetAction(Action):
    def __init__(self, lattice: tuple[int, ...], **couplings):
        super().__init__()
        self.lattice = lattice
        self.couplings = couplings

    def extra_repr(self) -> str:
        return ", ".join(
            f"{key}={val}"
            for key, val in (
                {"lattice": self.lattice} | self.couplings
            ).items()
        )


class PullbackAction(Action):
    def __init__(self, model: Model):
        super().__init__()
        self.register_module("model", model)

    def extra_repr(self) -> str:
        return self.model.extra_repr()

    @torch.no_grad()
    def forward(self, inputs: Tensor | Tensors) -> Tensor:
        outputs, log_det_jacobian = self.model.flow_forward(inputs)
        pullback = self.model.compute_target(outputs) - log_det_jacobian
        return pullback

    @torch.enable_grad()
    def grad(self, inputs: Tensor | Tensors) -> Tensor | Tensors:
        return self.model.grad_pullback(inputs)


class Hamiltonian(nn.Module, metaclass=ABCMeta):
    rng: torch.Generator | None = None

    @abstractmethod
    def forward(self, coords: Tensor | Tensors, momenta: Tensor | Tensors) -> Tensor: ...

    @abstractmethod
    def grad_wrt_coords(
        self, coords: Tensor | Tensors, momenta: Tensor | Tensors
    ) -> Tensor | Tensors: ...

    @abstractmethod
    def grad_wrt_momenta(
        self, coords: Tensor | Tensors, momenta: Tensor | Tensors
    ) -> Tensor | Tensors: ...

    @abstractmethod
    def sample_momenta(self, coords: Tensor | Tensors) -> Tensor | Tensors: ...



class SeparableHamiltonian(Hamiltonian):
    def __init__(self, potential: Action):
        super().__init__()
        self.register_module("potential", potential)

    @abstractmethod
    def kinetic(self, momenta: Tensor | Tensors) -> Tensor:
        ...

    def forward(self, coords: Tensor | Tensors, momenta: Tensor | Tensors) -> Tensor:
        return self.kinetic(momenta) + self.potential(coords)

    def grad_wrt_coords(
        self, coords: Tensor | Tensors, momenta: Tensor | Tensors
    ) -> Tensor | Tensors:
        return self.potential.grad(coords)

# NOTE: in practice are target hamiltonians always separable?
