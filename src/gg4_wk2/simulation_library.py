from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from tqdm import tqdm

from gg4_wk2.condition_library import ConditionLibrary
from gg4_wk2.simulation import Simulation


@dataclass(frozen=True)
class SimulationLibrary:
    simulations: tuple[Simulation, ...]

    @classmethod
    def from_condition_library(
        cls,
        library: ConditionLibrary,
        *,
        seed: int | None = None,
    ) -> SimulationLibrary:
        simulations: list[Simulation] = []

        with tqdm(total=len(library.conditions), desc="Running simulations") as pbar:
            for i, condition in enumerate(library.conditions):
                condition_seed = None if seed is None else seed + i
                latent_states, observations = condition.run(seed=condition_seed)
                simulations.append(
                    Simulation(
                        condition=condition,
                        latent_states=latent_states,
                        observations=observations,
                    )
                )
                pbar.update(1)

        return cls(simulations=tuple(simulations))

    def __len__(self) -> int:
        return len(self.simulations)

    def __getitem__(self, index: int) -> Simulation:
        return self.simulations[index]

    def __iter__(self) -> Iterator[Simulation]:
        return iter(self.simulations)
