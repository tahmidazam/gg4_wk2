from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseEstimator(ABC):
    @abstractmethod
    def estimate_latent_and_input(
        self, observation: np.ndarray, LatentDim: int, InputDim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class RandomEstimator(BaseEstimator):
    def estimate_latent_and_input(
        self, observation: np.ndarray, LatentDim: int, InputDim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Timepoints = (
            observation.shape[-2] if observation.ndim == 3 else observation.shape[0]
        )
        latent_states = np.random.rand(Timepoints, LatentDim)
        inputs = np.random.rand(Timepoints, InputDim)

        return latent_states, inputs
