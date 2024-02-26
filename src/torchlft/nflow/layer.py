from abc import ABCMeta, abstractmethod
from typing import Self, TypeAlias

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.Tensor


class Layer(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor | tuple[Tensor, ...], Tensor]: ...


class Composition(Layer):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.register_module("layers", nn.ModuleList(layers))

    def __iter__(self):
        return iter(self.layers)

    def forward(self, inputs: Tensor | tuple[Tensor, ...]):
        ldj_total = 0.0

        for layer in self:
            outputs, ldj = layer(inputs)
            ldj_total += ldj

            inputs = outputs

        return outputs, ldj_total
