from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gg4_wk2.validation import assert_shape


class Signal:
    __slots__ = ("T", "m", "data")

    def __init__(self, T: int, m: int, data: NDArray) -> None:
        data = np.asarray(data)
        assert_shape(data, (T, m), "data")
        self.T: int = T
        self.m: int = m
        self.data: NDArray[np.float64] = data.astype(np.float64)

    def __array__(self, dtype=None, copy=None) -> NDArray:
        arr = self.data
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __repr__(self) -> str:
        return f"Signal(T={self.T}, m={self.m}, data=\n{self.data!r})"

    def copy(self) -> Signal:
        return Signal(self.T, self.m, self.data.copy())

    def _check_compatible(self, other: Signal) -> None:
        if (self.T, self.m) != (other.T, other.m):
            raise ValueError(
                f"shape mismatch: {(self.T, self.m)} vs {(other.T, other.m)}"
            )

    def __add__(self, other: Signal) -> Signal:
        self._check_compatible(other)
        return Signal(self.T, self.m, self.data + other.data)

    def __radd__(self, other: Signal) -> Signal:
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: Signal) -> Signal:
        self._check_compatible(other)
        return Signal(self.T, self.m, self.data - other.data)

    def __neg__(self) -> Signal:
        return Signal(self.T, self.m, -self.data)

    def __mul__(self, scalar: float) -> Signal:
        return Signal(self.T, self.m, self.data * scalar)

    def __rmul__(self, scalar: float) -> Signal:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Signal:
        return Signal(self.T, self.m, self.data / scalar)

    def __getitem__(self, key):
        return self.data[key]
