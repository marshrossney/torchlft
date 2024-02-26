from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeAlias

import torch
from tqdm import trange

from torchlft.utils.torch import tuple_stack, dict_stack

# torch.set_default_dtype(torch.double)

Tensor: TypeAlias = torch.Tensor
Tensors: TypeAlias = tuple[Tensor, ...]


class SamplingAlgorithm(ABC):
    def __init__(self):
        self.rng = torch.Generator(device="cpu")

    def seed_rng(self, rng_seed: int | None = None) -> int:
        if rng_seed is not None:
            self.rng.manual_seed(rng_seed)
        else:
            rng_seed = self.rng.seed()
        return rng_seed

    @abstractmethod
    def init(self, seed: int | None = None) -> Tensor | Tensors: ...

    @abstractmethod
    def update(
        self, state: Tensor | Tensors
    ) -> tuple[Tensor | Tensors, Any]: ...

    def compute_stats(self, log: list[Any]):
        raise NotImplementedError


class Sampler(ABC):
    @abstractmethod
    def sample(self, context: Any = None): ...


class DefaultSampler(Sampler):
    def __init__(
        self,
        sample_size: int,
        thermalisation: int,
    ):
        self.sample_size = sample_size
        self.thermalisation = thermalisation

    def sample(self, algorithm):
        state = algorithm.init()

        with trange(self.thermalisation, desc="Thermalisation") as pbar:
            for step in pbar:
                state, _ = algorithm.update(state)

        sample_, stats = [], []

        with trange(self.sample_size, desc="Sampling") as pbar:
            for step in pbar:
                state, extra = algorithm.update(state)
                sample_.append(state.clone())
                stats.append(extra)

        try:
            stats = algorithm.compute_stats(stats)
        except NotImplementedError:
            stats = None

        # TODO: this isn't great
        if isinstance(state, Tensor):
            sample_ = torch.stack(sample_)
        elif isinstance(state, tuple):
            sample_ = tuple_stack(sample_)
        elif isinstance(state, dict):
            sample = dict_stack(sample_)

        return sample_, stats
