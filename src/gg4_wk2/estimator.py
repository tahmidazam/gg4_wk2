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
