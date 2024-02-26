"""
TODO: integrated autocorrelation and rejection histogram from https://github.com/marshrossney/torchlft/blob/archive/rewrite-2023/src/torchlft/metrics.py
"""

from math import exp
from random import random
from typing import TypeAlias

import torch


Tensor: TypeAlias = torch.Tensor


def _metropolis_acceptance(log_weights: Tensor) -> float:
    assert log_weights.dim() == 1
    log_weights = log_weights.tolist()
    current = log_weights.pop(0)

    idx = 0
    indices = []

    for proposal in log_weights:
        if proposal > current or random() < min(1, exp(proposal - current)):
            current = proposal
            idx += 1

        indices.append(idx)

    transitions = set(indices)
    transitions.discard(0)  # there was no transition *to* 0th state

    return len(transitions) / len(log_weights)


class LogWeightMetrics:
    def __init__(self):
        self._logw = []

    @staticmethod
    def metropolis_acceptance(log_weights: Tensor, dim: int = 1) -> Tensor:
        return torch.tensor(
            [_metropolis_acceptance(x) for x in log_weights],
            dtype=torch.float,
        )

    @staticmethod
    def effective_sample_size(log_weights: Tensor, dim: int = 1) -> Tensor:
        ess = torch.exp(
            log_weights.logsumexp(dim).mul(2)
            - log_weights.mul(2).logsumexp(dim)
        )
        ess /= log_weights.shape[dim]
        return ess

    def update(self, log_weights: Tensor):
        logw = (
            log_weights.squeeze().detach().to(device="cpu", dtype=torch.double)
        )
        assert logw.dim() == 1
        self._logw.append(logw)

    def compute(self):
        logw = torch.stack(self._logw)

        acc = self.metropolis_acceptance(logw)
        ess = self.effective_sample_size(logw)
        mlw = logw.mean(dim=1)
        vlw = logw.var(dim=1)

        return {
            "acc": acc,
            "ess": ess,
            "mlw": mlw,
            "vlw": vlw,
        }
