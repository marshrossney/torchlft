from itertools import chain
import math
from typing import TypeAlias

import torch

from torchlft.lattice.sample import SamplingAlgorithm
from torchlft.lattice.scalar.action import get_local_action
from torchlft.utils.lattice import build_neighbour_list


Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor


# NOTE: should I replace (lattice, couplings) with Target?
class RandomWalkMetropolis(SamplingAlgorithm):
    def __init__(
        self,
        lattice: tuple[int, ...],
        step_size: float,
        couplings: dict[str, float],
    ) -> None:
        super().__init__()
        self.lattice = lattice
        self.step_size = step_size
        self.couplings = couplings
        self.local_action = get_local_action(couplings)

        self.lattice_size = math.prod(lattice)
        self.neighbour_list = build_neighbour_list(lattice)

        self.rng = torch.Generator("cpu")

    @property
    def sweep_length(self) -> int:
        return self.lattice_size

    def init(self, rng_seed: int | None = None) -> None:

        if rng_seed is not None:
            self.rng.manual_seed(rng_seed)

        state = torch.empty(self.lattice).normal_(0, 1, generator=self.rng)

        # This is just a view of the original state
        flattened_state = state.view(-1)

        return flattened_state

    def _update(self, state, site_idx, δφ, u) -> tuple[Tensor, bool]:
        neighbour_idxs = self.neighbour_list[site_idx]

        φ_old, *φ_neighbours = state[[site_idx, *neighbour_idxs]]
        φ_new = φ_old + δφ

        S_old = self.local_action(φ_old, φ_neighbours)
        S_new = self.local_action(φ_new, φ_neighbours)

        if torch.exp(S_new - S_old) > u:
            state[site_idx] = φ_new
            return True
        else:
            return False

    def update(self, state: Tensor) -> tuple[Tensor, list[bool]]:
        D = self.lattice_size
        site_indices = torch.randint(0, D, [D], generator=self.rng).tolist()
        perturbations = torch.randn(D, generator=self.rng) * self.step_size
        uniform_numbers = torch.rand(D, generator=self.rng)

        accepted = []

        for site_idx, δφ, u in zip(
            site_indices, perturbations, uniform_numbers
        ):
            accepted.append(self._update(state, site_idx, δφ, u))

        return state, accepted

    def compute_stats(self, history: list[list[bool]]) -> dict[str, float]:
        history = list(chain(*history))
        acceptance = sum(history) / len(history)
        return {"acceptance": acceptance}


def main():
    from torchlft.lattice.sample import DefaultSampler

    sampler = DefaultSampler(1000, 100, 1234567)
    alg = RandomWalkMetropolis([6, 6], 0.5, {"β": 0.5, "λ": 0.5})
    configs, stats = sampler.sample(alg)

    print(configs.shape)
    print(stats)


if __name__ == "__main__":
    main()
