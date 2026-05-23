from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from gg4_wk2.simulator import LinearGaussianModel
from gg4_wk2.waveform import (
    Constant,
    Oscillation,
    Pulse,
    Ramp,
    Waveform,
    WaveformKind,
    WhiteNoise,
)


class BaseSampler[T](ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, *args, **kwargs) -> T: ...


class LinearGaussianModelSampler(BaseSampler[LinearGaussianModel]):
    __slots__ = (
        "state_dim_range",
        "input_dim_range",
        "obs_dim_range",
        "rho_range",
        "log_sQ_range",
        "log_sR_range",
        "max_retries",
    )

    def __init__(
        self,
        *,
        state_dim_range: tuple[int, int] = (2, 6),
        input_dim_range: tuple[int, int] = (1, 4),
        obs_dim_range: tuple[int, int] = (2, 6),
        rho_range: tuple[float, float] = (0.1, 0.95),
        log_sQ_range: tuple[float, float] = (-2.0, 1.0),
        log_sR_range: tuple[float, float] = (-2.0, 1.0),
        max_retries: int = 16,
    ) -> None:
        self.state_dim_range = state_dim_range
        self.input_dim_range = input_dim_range
        self.obs_dim_range = obs_dim_range
        self.rho_range = rho_range
        self.log_sQ_range = log_sQ_range
        self.log_sR_range = log_sR_range
        self.max_retries = max_retries

    def sample(self, rng: np.random.Generator) -> LinearGaussianModel:
        for _ in range(self.max_retries):
            n = int(rng.integers(self.state_dim_range[0], self.state_dim_range[1] + 1))
            m = int(rng.integers(self.input_dim_range[0], self.input_dim_range[1] + 1))
            len = int(rng.integers(self.obs_dim_range[0], self.obs_dim_range[1] + 1))

            rho = float(rng.uniform(*self.rho_range))
            log_sQ = float(rng.uniform(*self.log_sQ_range))
            log_sR = float(rng.uniform(*self.log_sR_range))

            A_raw = rng.standard_normal((n, n))
            rho_raw = float(np.max(np.abs(np.linalg.eigvals(A_raw))))
            if rho_raw < 1e-12:
                continue
            A = (rho / rho_raw) * A_raw

            B = rng.standard_normal((n, m))
            C = rng.standard_normal((len, n))

            sQ = 10.0**log_sQ
            sR = 10.0**log_sR
            M_Q = rng.standard_normal((n, n))
            M_R = rng.standard_normal((len, len))
            Q = (sQ**2) * (M_Q @ M_Q.T)
            R = (sR**2) * (M_R @ M_R.T)

            try:
                return LinearGaussianModel(A=A, B=B, C=C, Q=Q, R=R)
            except ValueError:
                continue

        raise RuntimeError(
            f"failed to sample a valid LinearGaussianModel in {self.max_retries} retries"
        )


class IntegerSampler(BaseSampler[int]):
    __slots__ = ("low", "high", "log")

    def __init__(self, *, low: int, high: int, log: bool = False) -> None:
        if log and low < 1:
            raise ValueError(f"log=True requires low >= 1, got low={low}")
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        self.low = low
        self.high = high
        self.log = log

    def sample(self, rng: np.random.Generator) -> int:
        if self.log:
            return int(
                round(10.0 ** rng.uniform(math.log10(self.low), math.log10(self.high)))
            )
        return int(rng.integers(self.low, self.high + 1))


_DEFAULT_KIND_WEIGHTS: dict[WaveformKind, float] = {kind: 1.0 for kind in WaveformKind}


class WaveformSampler(BaseSampler[Waveform]):
    __slots__ = (
        "kind_weights",
        "log_amplitude_range",
        "log_pulse_duration_frac_range",
        "log_frequency_range",
        "log_noise_scale_range",
    )

    def __init__(
        self,
        *,
        kind_weights: dict[WaveformKind, float] | None = None,
        log_amplitude_range: tuple[float, float] = (-1.0, 1.0),
        log_pulse_duration_frac_range: tuple[float, float] = (-2.0, math.log10(0.25)),
        log_frequency_range: tuple[float, float] = (-3.0, math.log10(0.5)),
        log_noise_scale_range: tuple[float, float] = (-2.0, 1.0),
    ) -> None:
        self.kind_weights = (
            dict(_DEFAULT_KIND_WEIGHTS) if kind_weights is None else dict(kind_weights)
        )
        self.log_amplitude_range = log_amplitude_range
        self.log_pulse_duration_frac_range = log_pulse_duration_frac_range
        self.log_frequency_range = log_frequency_range
        self.log_noise_scale_range = log_noise_scale_range

    def sample(
        self, rng: np.random.Generator, time_steps: int, input_dim: int
    ) -> Waveform:
        kinds = list(self.kind_weights.keys())
        weights = np.array([self.kind_weights[k] for k in kinds], dtype=np.float64)
        weights /= weights.sum()
        kind = kinds[int(rng.choice(len(kinds), p=weights))]

        if kind is WaveformKind.CONSTANT:
            return Constant()

        if kind is WaveformKind.PULSE:
            duration_frac = 10.0 ** rng.uniform(*self.log_pulse_duration_frac_range)
            duration = max(1, int(round(time_steps * duration_frac)))
            duration = min(duration, time_steps)
            onset = int(rng.integers(0, time_steps - duration + 1))
            return Pulse(
                onset=onset,
                duration=duration,
                amplitude=self._signed_amplitude(rng),
                channels=self._channels(rng, input_dim),
            )

        if kind is WaveformKind.OSCILLATION:
            f_lo, f_hi = self.log_frequency_range
            f_lo = max(f_lo, math.log10(1.0 / time_steps))
            log_f = rng.uniform(f_lo, f_hi)
            return Oscillation(
                frequency=float(10.0**log_f),
                amplitude=abs(self._signed_amplitude(rng)),
                phase=float(rng.uniform(0, 2 * np.pi)),
                channels=self._channels(rng, input_dim),
            )

        if kind is WaveformKind.RAMP:
            scale = 10.0 ** rng.uniform(*self.log_amplitude_range)
            return Ramp(
                start=float(rng.uniform(-1, 1) * scale),
                end=float(rng.uniform(-1, 1) * scale),
                channels=self._channels(rng, input_dim),
            )

        if kind is WaveformKind.WHITE_NOISE:
            return WhiteNoise(
                scale=10.0 ** rng.uniform(*self.log_noise_scale_range),
                seed=int(rng.integers(0, 2**31 - 1)),
            )

    def _channels(self, rng: np.random.Generator, m: int) -> tuple[int, ...]:
        size = int(rng.integers(1, m + 1))
        idx = rng.choice(m, size=size, replace=False)
        return tuple(sorted(int(i) for i in idx))

    def _signed_amplitude(self, rng: np.random.Generator) -> float:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return sign * (10.0 ** rng.uniform(*self.log_amplitude_range))
