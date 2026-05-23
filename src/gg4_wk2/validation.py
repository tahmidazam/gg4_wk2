from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def assert_2d(M: NDArray, name: str) -> None:
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got {M.ndim}D array")


def assert_square(M: NDArray, name: str) -> None:
    assert_2d(M, name)
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got shape {M.shape}")


def assert_shape(M: NDArray, expected: tuple[int, ...], name: str) -> None:
    if M.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {M.shape}")


def assert_symmetric(M: NDArray, name: str, *, atol: float = 1e-8) -> None:
    assert_square(M, name)
    if not np.allclose(M, M.T, atol=atol):
        raise ValueError(f"{name} must be symmetric (atol={atol})")


def assert_psd(M: NDArray, name: str, *, atol: float = 1e-10) -> None:
    assert_square(M, name)
    if np.any(np.linalg.eigvalsh(M) < -atol):
        raise ValueError(f"{name} must be positive semi-definite (atol={atol})")
