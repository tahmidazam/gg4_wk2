from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gg4_wk2.validation import (
    assert_2d,
    assert_psd,
    assert_shape,
    assert_square,
    assert_symmetric,
)


class LinearGaussianModel:
    __slots__ = ("A", "B", "C", "Q", "R")

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
    ) -> None:
        A, B, C, Q, R = (np.asarray(m, dtype=np.float64) for m in (A, B, C, Q, R))

        assert_square(A, "A")
        state_dim = A.shape[0]

        assert_2d(B, "B")
        assert_shape(B, (state_dim, B.shape[1]), "B")

        assert_2d(C, "C")
        assert_shape(C, (C.shape[0], state_dim), "C")
        obs_dim = C.shape[0]

        assert_symmetric(Q, "Q")
        assert_psd(Q, "Q")
        assert_shape(Q, (state_dim, state_dim), "Q")

        assert_symmetric(R, "R")
        assert_psd(R, "R")
        assert_shape(R, (obs_dim, obs_dim), "R")

        self.A: NDArray[np.float64] = A
        self.B: NDArray[np.float64] = B
        self.C: NDArray[np.float64] = C
        self.Q: NDArray[np.float64] = Q
        self.R: NDArray[np.float64] = R

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def input_dim(self) -> int:
        return self.B.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.C.shape[0]
