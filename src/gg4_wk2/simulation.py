from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gg4_wk2.condition import Condition
from gg4_wk2.observations import Observations


@dataclass(frozen=True)
class Simulation:
    condition: Condition
    latent_states: NDArray[np.float64]
    observations: Observations

    def __init__(
        self,
        condition: Condition,
        latent_states: ArrayLike,
        observations: Observations,
    ) -> None:
        arr = np.array(latent_states, dtype=np.float64, copy=True)

        if arr.ndim != 3:
            raise ValueError(
                "latent_states must have shape "
                "(n_trials, n_timesteps, state_dim); "
                f"got array with shape {arr.shape}"
            )

        arr.setflags(write=False)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "latent_states", arr)
        object.__setattr__(self, "observations", observations)
