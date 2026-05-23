from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gg4_wk2.simulator import LinearGaussianModel, LinearGaussianSimulator
from gg4_wk2.waveform import Waveform


@dataclass(frozen=True)
class Condition:
    model: LinearGaussianModel
    x_0: Waveform
    input: Waveform
    time_steps: int
    trials: int

    _CONDITION_FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "log10_time_steps",
        "trials",
        "log10_x0_norm",
        "log10_input_rms",
        "log10_excitation_to_process_noise_ratio",
        "input_kind_id",
    )

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        LinearGaussianModel.FEATURE_NAMES + _CONDITION_FEATURE_NAMES
    )

    @property
    def initial_state(self) -> NDArray[np.float64]:
        return self.x_0.to_signal(1, self.model.state_dim).data[0]

    def run(
        self, *, seed: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        u = self.input.to_signal(self.time_steps, self.model.input_dim).data
        sim = LinearGaussianSimulator(self.model, seed=seed)
        return sim.run(self.initial_state, u, self.time_steps, self.trials)

    @property
    def log10_time_steps(self) -> float:
        return math.log10(self.time_steps)

    @property
    def log10_x0_norm(self) -> float:
        x0_norm = float(np.linalg.norm(self.initial_state))
        return math.log10(x0_norm) if x0_norm > 0 else float("nan")

    @property
    def log10_input_rms(self) -> float:
        rms = self.input.root_mean_sqare(self.time_steps, self.model.input_dim)
        return math.log10(rms) if rms > 0 else float("nan")

    @property
    def log10_excitation_to_process_noise_ratio(self) -> float:
        rms = self.input.root_mean_sqare(self.time_steps, self.model.input_dim)
        Q_trace = float(np.trace(self.model.Q))
        B_frob_sq = float(np.sum(self.model.B**2))
        if rms > 0 and Q_trace > 0:
            return math.log10((rms**2 * B_frob_sq) / Q_trace)
        return float("nan")

    @property
    def input_kind_id(self) -> float:
        return float(self.input.KIND.id)

    @property
    def features(self) -> dict[str, float]:
        d = self.model.features
        d.update(
            {name: float(getattr(self, name)) for name in self._CONDITION_FEATURE_NAMES}
        )
        return d
