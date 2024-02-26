from math import isclose
from typing import Self, TypeAlias

import torch
import torch.nn as nn

from torchlft.lattice.action import Hamiltonian, SeparableHamiltonian

from torchlft.lattice.sample import SamplingAlgorithm
from torchlft.lattice.scalar.action import Phi4Action
from torchlft.utils.torch import dict_concat

Tensor: TypeAlias = torch.Tensor


class HamiltonianGaussianMomenta(SeparableHamiltonian):
    def kinetic(self, p: Tensor) -> Tensor:
        return 0.5 * p.pow(2).flatten(1).sum(1, keepdim=True)

    def grad_wrt_momenta(self, x: Tensor, p: Tensor) -> Tensor:
        return p.clone()

    def sample_momenta(self, x0: Tensor) -> Tensor:
        p0 = torch.empty_like(x0).normal_(generator=self.rng)
        return p0


class LeapfrogIntegrator:
    def __init__(
        self,
        velocity_func,
        force_func,
        step_size: float,
        traj_length: float = 1.0,
    ):

        # TODO: prioritise traj_length or step_size?
        n_steps = max(1, round(traj_length / abs(step_size)))
        real_traj_length = n_steps * step_size
        if not isclose(real_traj_length, traj_length):
            print("baddies")  # TODO

        self.velocity_func = velocity_func
        self.force_func = force_func

        self.step_size = step_size
        self.n_steps = n_steps
        self.traj_length = traj_length

    @classmethod
    def from_hamiltonian(cls, hamiltonian: Hamiltonian, **kwargs) -> Self:
        return cls(
            velocity_func=hamiltonian.grad_wrt_momenta,
            force_func=lambda *inputs: hamiltonian.grad_wrt_coords(
                *inputs
            ).negative(),
            **kwargs,
        )

    def on_step(self, x: Tensor, p: Tensor, t: float) -> None:
        pass

    def _integrate(self, x, p, t, inverse):

        x = x.clone()
        p = p.clone()
        ε = -self.step_size if inverse else self.step_size

        F = self.force_func(x, p)

        for _ in range(self.n_steps):
            self.on_step(x, p, t)

            # NOTE: avoid in-place here in case p stored in on_step_func
            p = p + (ε / 2) * F

            v = self.velocity_func(x, p)

            x = x + ε * v

            F = self.force_func(x, p)

            p += (ε / 2) * F

            t += ε

        return x, p, t

    def __call__(
        self, coords: Tensor, momenta: Tensor
    ) -> tuple[Tensor, Tensor, float]:
        return self._integrate(coords, momenta, t=0, inverse=False)

    def inverse(
        self, coords: Tensor, momenta: Tensor
    ) -> tuple[Tensor, Tensor, float]:
        return self._integrate(
            coords, momenta, t=self.traj_length, inverse=True
        )


class HybridMonteCarlo(SamplingAlgorithm):

    def __init__(
        self,
        lattice,
        hamiltonian: Hamiltonian,
        step_size: float,
        traj_length: float = 1.0,
        n_replica: int = 1,
        flow_hamiltonian: Hamiltonian | None = None,
    ):
        self.lattice = lattice

        self.hamiltonian = hamiltonian
        self.n_replica = n_replica

        if flow_hamiltonian is None:
            flow_hamiltonian = hamiltonian

        self.integrator = LeapfrogIntegrator.from_hamiltonian(
            flow_hamiltonian,
            step_size=step_size,
            traj_length=traj_length,
        )

        self.rng = torch.Generator("cpu")
        self.hamiltonian.rng = self.rng

    def seed_rng(self, rng_seed: int | None = None) -> int:
        if rng_seed is not None:
            self.rng.manual_seed(rng_seed)
        else:
            rng_seed = self.rng.seed()
        return rng_seed

    def init(self):

        # TODO shape is complicated: depends on flow!
        state = torch.empty(self.n_replica, *self.lattice).normal_(
            0, 1, generator=self.rng
        )

        return state  # .flatten(1)

    def update(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:

        φ0 = state
        ω0 = self.hamiltonian.sample_momenta(φ0)

        H0 = self.hamiltonian(φ0, ω0)

        φt, ωt, t = self.integrator(φ0, ω0)

        Ht = self.hamiltonian(φt, ωt)

        unif = torch.rand(Ht.shape, generator=self.rng)
        accepted = torch.exp(Ht - H0) > unif

        φt = torch.where(accepted, φt, φ0)

        logs = {"ΔH": H0 - Ht, "accepted": accepted}

        return φt, logs

    def compute_stats(self, logs: list[dict[str, Tensor]]):
        # TODO: actually useful stats, mean plus error

        logs = dict_concat(logs)

        ΔH = logs["ΔH"]
        stat = torch.exp(ΔH).mean() - 1  # TODO

        acceptance = logs["accepted"]
        print("acc shape: ", acceptance.shape)
        acceptance = acceptance.float().mean()

        return {"zero_stat": stat, "acceptance": acceptance}


def main():
    from torchlft.lattice.sample import DefaultSampler

    sampler = DefaultSampler(1000, 100)
    lattice = [6, 6]
    hamiltonian = HamiltonianGaussianMomenta(
        Phi4Action(lattice=lattice, β=0.5, λ=0.5)
    )
    alg = HybridMonteCarlo(lattice, hamiltonian, step_size=0.05)
    configs, stats = sampler.sample(alg)

    print("configs shape: ", configs.shape)
    print(stats)


if __name__ == "__main__":
    main()
