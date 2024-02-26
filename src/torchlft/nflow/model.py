from abc import ABCMeta, abstractmethod
from math import exp
from random import random
from typing import NamedTuple, TypeAlias

import torch
import torch.nn as nn

from torchlft.utils.torch import tuple_concat


Tensor: TypeAlias = torch.Tensor
Tensors: TypeAlias = tuple[Tensor, ...]


def metropolis_hastings(log_weights: Tensor):
    assert log_weights.squeeze().dim() == 1

    log_weights = log_weights.squeeze().tolist()

    current = log_weights.pop(0)

    idx = 0
    indices = []

    for proposal in log_weights:
        if (proposal > current) or (
            random() < min(1, exp(proposal - current))
        ):
            current = proposal
            idx += 1

        indices.append(idx)

    indices = torch.tensor(indices, dtype=torch.long)

    return indices


class Model(nn.Module, metaclass=ABCMeta):
    class Fields(NamedTuple):
        inputs: Tensor | Tensors
        outputs: Tensor | Tensors

    class Actions(NamedTuple):
        base: Tensor
        target: Tensor
        pushforward: Tensor
        pullback: Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy_buffer", torch.tensor(0.0))

    @property
    def device(self) -> torch.device:
        return self._dummy_buffer.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dummy_buffer.dtype

    @property
    def parameter_count(self) -> int:
        return sum(tensor.numel() for tensor in self.parameters())

    def forward(
        self, batch_size: int
    ) -> tuple["Model.Fields", "Model.Actions"]:
        inputs, base_action = self.sample_base(batch_size)
        outputs, log_det_jacobian = self.flow_forward(inputs)
        target_action = self.compute_target(outputs)

        pushforward = base_action + log_det_jacobian
        pullback = target_action - log_det_jacobian

        fields = self.Fields(inputs, outputs)
        actions = self.Actions(
            base_action, target_action, pushforward, pullback
        )

        return fields, actions

    @torch.no_grad()
    def weighted_sample(
        self, batch_size: int, n_batches: int = 1
    ) -> tuple[Tensor | Tensors, Tensor]:
        sample = []
        for _ in range(n_batches):
            fields, actions = self(batch_size)
            log_weights = actions.pushforward - actions.target
            sample.append([fields.outputs, log_weights])

        return tuple_concat(sample)

    @torch.no_grad()
    def metropolis_sample(
        self, batch_size: int, n_batches: int = 1
    ) -> tuple[Tensor, Tensor]:
        outputs, log_weights = self.weighted_sample(batch_size, n_batches)
        indices = metropolis_hastings(log_weights)
        return (outputs, indices)

    @abstractmethod
    def flow_forward(
        self, inputs: Tensor | Tensors
    ) -> tuple[Tensor | Tensors, Tensor]: ...

    @abstractmethod
    def sample_base(
        self,
        batch_size: int,
    ) -> tuple[Tensor | Tensors, Tensor]: ...

    @abstractmethod
    def compute_target(
        self,
        inputs: Tensor | Tensors,
    ) -> Tensor: ...

    def grad_pullback(self, inputs: Tensor | Tensors) -> Tensor:
        raise NotImplementedError
