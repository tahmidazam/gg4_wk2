from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gg4_wk2.simulator import LinearGaussianModel, LinearGaussianSimulator
from gg4_wk2.waveform import Waveform

CONDITION_SCALAR_PROPERTY_NAMES: tuple[str, ...] = (
    "log10_time_steps",
    "trials",
    "log10_x0_norm",
    "log10_input_rms",
    "log10_excitation_to_process_noise_ratio",
    "input_kind_id",
)


@dataclass(frozen=True)
class Condition:
    model: LinearGaussianModel
    x_0: Waveform
    input: Waveform
    time_steps: int
    trials: int

    SCALAR_PROPERTY_NAMES: ClassVar[tuple[str, ...]] = (
        LinearGaussianModel.SCALAR_PROPERTY_NAMES + CONDITION_SCALAR_PROPERTY_NAMES
    )

    def initial_state(self) -> NDArray[np.float64]:
        return self.x_0.to_signal(1, self.model.state_dim).data[0]

    def run(
        self, *, seed: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        u = self.input.to_signal(self.time_steps, self.model.input_dim).data
        sim = LinearGaussianSimulator(self.model, seed=seed)
        return sim.run(self.initial_state(), u, self.time_steps, self.trials)

    @property
    def scalar_property_vector(self) -> NDArray[np.float64]:
        model_vec = self.model.scalar_property_vector
        m = self.model.input_dim
        T = self.time_steps

        x0_norm = float(np.linalg.norm(self.initial_state()))
        input_rms = self.input.root_mean_sqare(T, m)
        Q_trace = float(np.trace(self.model.Q))
        B_frob_sq = float(np.sum(self.model.B**2))

        log10_T = math.log10(T)
        log10_x0_norm = math.log10(x0_norm) if x0_norm > 0 else float("nan")
        log10_input_rms = math.log10(input_rms) if input_rms > 0 else float("nan")
        if input_rms > 0 and Q_trace > 0:
            log10_excitation = math.log10((input_rms**2 * B_frob_sq) / Q_trace)
        else:
            log10_excitation = float("nan")
        kind_id = float(self.input.KIND.id)

        extras = np.array(
            [
                log10_T,
                float(self.trials),
                log10_x0_norm,
                log10_input_rms,
                log10_excitation,
                kind_id,
            ],
            dtype=np.float64,
        )
        return np.concatenate([model_vec, extras])
