from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gg4_wk2.observations import Observations
from gg4_wk2.model import LinearGaussianModel
from gg4_wk2.validation import (
    assert_shape,
)


class BaseSimulator(ABC):
    @abstractmethod
    def run(
        self,
        x_0: NDArray[np.float64],
        u: NDArray[np.float64],
        time_steps: int,
        trials: int = 1,
    ) -> tuple[NDArray[np.float64], Observations]:
        pass


class LinearGaussianSimulator(BaseSimulator):
    __slots__ = ("model", "rng")

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        seed: int | None = None,
    ) -> None:
        self.model: LinearGaussianModel = model
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def run(
        self,
        x_0: ArrayLike,
        u: ArrayLike,
        time_steps: int,
        trials: int = 1,
    ) -> tuple[NDArray[np.float64], Observations]:
        x_0 = np.asarray(x_0, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        if x_0.ndim == 1:
            assert_shape(x_0, (self.model.state_dim,), "x_0")
        else:
            assert_shape(x_0, (trials, self.model.state_dim), "x_0")
        assert_shape(u, (time_steps, self.model.input_dim), "u")

        w = self.rng.multivariate_normal(
            np.zeros(self.model.state_dim), self.model.Q, size=(trials, time_steps)
        )
        v = self.rng.multivariate_normal(
            np.zeros(self.model.obs_dim), self.model.R, size=(trials, time_steps)
        )

        x = np.empty((trials, time_steps, self.model.state_dim))
        x[:, 0] = x_0
        for t in range(1, time_steps):
            x[:, t] = (
                x[:, t - 1] @ self.model.A.T + u[t - 1] @ self.model.B.T + w[:, t - 1]
            )

        y = x @ self.model.C.T + v

        return x, Observations(y)
