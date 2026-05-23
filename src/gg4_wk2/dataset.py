from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class Dataset:
    observations: NDArray[np.float64]

    def __init__(self, observations: ArrayLike) -> None:
        arr = np.asarray(observations, dtype=np.float64)

        if arr.ndim != 3:
            raise ValueError(
                "observations must have shape "
                "(n_trials, n_timesteps, n_neurons); "
                f"got array with shape {arr.shape}"
            )

        arr = np.array(arr, dtype=np.float64, copy=True)
        arr.setflags(write=False)
        object.__setattr__(self, "observations", arr)

    @property
    def n_trials(self) -> int:
        return self.observations.shape[0]

    @property
    def n_timesteps(self) -> int:
        return self.observations.shape[1]

    @property
    def n_neurons(self) -> int:
        return self.observations.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.observations.shape

    def trial(self, index: int) -> NDArray[np.float64]:
        return self.observations[index]

    def neuron(self, index: int) -> NDArray[np.float64]:
        return self.observations[:, :, index]

    def timestep(self, index: int) -> NDArray[np.float64]:
        return self.observations[:, index, :]

    def __len__(self) -> int:
        return self.n_trials

    def __array__(self) -> NDArray[np.float64]:
        return self.observations

    def __repr__(self) -> str:
        return (
            "Dataset("
            f"shape={self.shape}, "
            f"n_trials={self.n_trials}, "
            f"n_timesteps={self.n_timesteps}, "
            f"n_neurons={self.n_neurons}"
            ")"
        )
