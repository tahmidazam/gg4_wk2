from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np

from gg4_wk2.signal import Signal


class WaveformKind(StrEnum):
    CONSTANT = "constant"
    PULSE = "pulse"
    OSCILLATION = "oscillation"
    RAMP = "ramp"
    WHITE_NOISE = "white_noise"

    @property
    def id(self) -> int:
        return list(WaveformKind).index(self)


_WAVEFORM_REGISTRY: dict[WaveformKind, type[Waveform]] = {}


@dataclass(frozen=True)
class Waveform(ABC):
    KIND: ClassVar[WaveformKind]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "KIND" in cls.__dict__:
            _WAVEFORM_REGISTRY[cls.KIND] = cls

    @abstractmethod
    def to_signal(self, T: int, m: int) -> Signal: ...

    @abstractmethod
    def root_mean_sqare(self, T: int, m: int) -> float: ...

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Waveform:
        d = dict(data)
        kind = WaveformKind(d.pop("kind"))
        subclass = _WAVEFORM_REGISTRY[kind]
        if "channels" in d and d["channels"] is not None:
            d["channels"] = tuple(d["channels"])
        return subclass(**d)


def _resolve_channels(channels: tuple[int, ...] | None, m: int) -> list[int]:
    return list(range(m)) if channels is None else list(channels)


@dataclass(frozen=True)
class Constant(Waveform):
    KIND: ClassVar[WaveformKind] = WaveformKind.CONSTANT
    value: float = 0.0

    def to_signal(self, T: int, m: int) -> Signal:
        return Signal(T, m, np.full((T, m), self.value, dtype=np.float64))

    def root_mean_sqare(self, T: int, m: int) -> float:
        return abs(self.value)


@dataclass(frozen=True)
class Pulse(Waveform):
    KIND: ClassVar[WaveformKind] = WaveformKind.PULSE
    onset: int
    duration: int
    amplitude: float
    channels: tuple[int, ...] | None = None

    def to_signal(self, T: int, m: int) -> Signal:
        x = np.zeros((T, m), dtype=np.float64)
        x[
            self.onset : self.onset + self.duration, _resolve_channels(self.channels, m)
        ] = self.amplitude
        return Signal(T, m, x)

    def root_mean_sqare(self, T: int, m: int) -> float:
        k = m if self.channels is None else len(self.channels)
        return abs(self.amplitude) * math.sqrt(self.duration * k / (T * m))


@dataclass(frozen=True)
class Oscillation(Waveform):
    KIND: ClassVar[WaveformKind] = WaveformKind.OSCILLATION
    frequency: float
    amplitude: float = 1.0
    phase: float = 0.0
    channels: tuple[int, ...] | None = None

    def to_signal(self, T: int, m: int) -> Signal:
        t = np.arange(T)
        wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        x = np.zeros((T, m), dtype=np.float64)
        x[:, _resolve_channels(self.channels, m)] = wave[:, None]
        return Signal(T, m, x)

    def root_mean_sqare(self, T: int, m: int) -> float:
        k = m if self.channels is None else len(self.channels)
        return abs(self.amplitude) * math.sqrt(k / m / 2.0)


@dataclass(frozen=True)
class Ramp(Waveform):
    KIND: ClassVar[WaveformKind] = WaveformKind.RAMP
    start: float
    end: float
    channels: tuple[int, ...] | None = None

    def to_signal(self, T: int, m: int) -> Signal:
        line = np.linspace(self.start, self.end, T)
        x = np.zeros((T, m), dtype=np.float64)
        x[:, _resolve_channels(self.channels, m)] = line[:, None]
        return Signal(T, m, x)

    def root_mean_sqare(self, T: int, m: int) -> float:
        k = m if self.channels is None else len(self.channels)
        s, e = self.start, self.end
        return math.sqrt(k / m * (s * s + s * e + e * e) / 3.0)


@dataclass(frozen=True)
class WhiteNoise(Waveform):
    KIND: ClassVar[WaveformKind] = WaveformKind.WHITE_NOISE
    scale: float
    seed: int | None = None

    def to_signal(self, T: int, m: int) -> Signal:
        rng = np.random.default_rng(self.seed)
        return Signal(T, m, rng.normal(0.0, self.scale, (T, m)))

    def root_mean_sqare(self, T: int, m: int) -> float:
        return abs(self.scale)
