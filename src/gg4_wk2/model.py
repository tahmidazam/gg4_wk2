from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from gg4_wk2.validation import (
    assert_2d,
    assert_psd,
    assert_shape,
    assert_square,
    assert_symmetric,
)


def _effective_rank(eigs: NDArray[np.float64]) -> np.float64:
    pos = eigs[eigs > 0]
    if pos.size == 0:
        return np.float64(0.0)
    p = pos / pos.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


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

        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def input_dim(self) -> int:
        return self.B.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.C.shape[0]

    @property
    def A_eigenvalues(self) -> NDArray[np.complex128]:
        return np.linalg.eigvals(self.A)

    @property
    def spectral_radius_A(self) -> np.float64:
        return np.max(np.abs(self.A_eigenvalues))

    @property
    def is_schur_stable(self) -> bool:
        return bool(self.spectral_radius_A < 1.0)

    @property
    def spectral_gap_A(self) -> np.float64:
        moduli = np.sort(np.abs(self.A_eigenvalues))[::-1]
        if moduli.size < 2:
            return np.float64(np.inf)
        return moduli[0] - moduli[1]

    @property
    def dominant_mode_time_constant(self) -> np.float64:
        rho = self.spectral_radius_A
        if rho >= 1.0:
            return np.float64(np.inf)
        if rho == 0.0:
            return np.float64(0.0)
        return np.float64(-1.0 / np.log(rho))

    @property
    def dominant_mode_damping_ratio(self) -> np.float64:
        eigs = self.A_eigenvalues
        lam = eigs[np.argmax(np.abs(eigs))]
        if lam == 0:
            return np.float64(1.0)
        s = np.log(lam)
        if np.abs(s) == 0:
            return np.float64(1.0)
        return np.float64(-np.real(s) / np.abs(s))

    @property
    def dominant_mode_oscillation_frequency(self) -> np.float64:
        eigs = self.A_eigenvalues
        lam = eigs[np.argmax(np.abs(eigs))]
        if lam == 0:
            return np.float64(0.0)
        return np.float64(np.abs(np.imag(np.log(lam))))

    @property
    def henrici_non_normality_index(self) -> np.float64:
        frob_sq = np.sum(self.A**2)
        if frob_sq == 0:
            return np.float64(0.0)
        eig_energy = np.sum(np.abs(self.A_eigenvalues) ** 2)
        return np.float64(max(0.0, (frob_sq - eig_energy) / frob_sq))

    def _assert_schur_stable(self) -> None:
        if not self.is_schur_stable:
            raise ValueError(
                f"Gramian undefined as A is not Schur stable "
                f"(spectral radius = {self.spectral_radius_A:.4f})."
            )

    @property
    def W_c(self) -> NDArray[np.float64]:
        self._assert_schur_stable()
        return solve_discrete_lyapunov(self.A, self.B @ self.B.T)

    @property
    def W_o(self) -> NDArray[np.float64]:
        self._assert_schur_stable()
        return solve_discrete_lyapunov(self.A.T, self.C.T @ self.C)

    @property
    def controllability_gramian_eigenvalues(self) -> NDArray[np.float64]:
        return np.linalg.eigvalsh(self.W_c).astype(np.float64)

    @property
    def controllability_min_eigenvalue(self) -> np.float64:
        return np.float64(self.controllability_gramian_eigenvalues.min())

    @property
    def controllability_trace(self) -> np.float64:
        return np.float64(np.trace(self.W_c))

    @property
    def controllability_log_det(self) -> np.float64:
        sign, logdet = np.linalg.slogdet(self.W_c)
        if sign <= 0:
            return np.float64(-np.inf)
        return np.float64(logdet)

    @property
    def controllability_condition_number(self) -> np.float64:
        eigs = self.controllability_gramian_eigenvalues
        if eigs.min() <= 0:
            return np.float64(np.inf)
        return np.float64(eigs.max() / eigs.min())

    @property
    def controllability_effective_rank(self) -> np.float64:
        return _effective_rank(self.controllability_gramian_eigenvalues)

    @property
    def observability_gramian_eigenvalues(self) -> NDArray[np.float64]:
        return np.linalg.eigvalsh(self.W_o).astype(np.float64)

    @property
    def observability_min_eigenvalue(self) -> np.float64:
        return np.float64(self.observability_gramian_eigenvalues.min())

    @property
    def observability_trace(self) -> np.float64:
        return np.float64(np.trace(self.W_o))

    @property
    def observability_log_det(self) -> np.float64:
        sign, logdet = np.linalg.slogdet(self.W_o)
        if sign <= 0:
            return np.float64(-np.inf)
        return np.float64(logdet)

    @property
    def observability_condition_number(self) -> np.float64:
        eigs = self.observability_gramian_eigenvalues
        if eigs.min() <= 0:
            return np.float64(np.inf)
        return np.float64(eigs.max() / eigs.min())

    @property
    def observability_effective_rank(self) -> np.float64:
        return _effective_rank(self.observability_gramian_eigenvalues)

    @property
    def hankel_singular_values(self) -> NDArray[np.float64]:
        eigs = np.linalg.eigvals(self.W_c @ self.W_o).real
        eigs = np.clip(eigs, 0.0, None)
        return np.sort(np.sqrt(eigs))[::-1]

    @property
    def hankel_norm(self) -> np.float64:
        return np.float64(self.hankel_singular_values[0])

    @property
    def hankel_min_singular_value(self) -> np.float64:
        return np.float64(self.hankel_singular_values[-1])

    @property
    def hankel_condition_number(self) -> np.float64:
        hsv = self.hankel_singular_values
        if hsv[-1] <= 0:
            return np.float64(np.inf)
        return np.float64(hsv[0] / hsv[-1])

    @property
    def hankel_effective_rank(self) -> np.float64:
        return _effective_rank(self.hankel_singular_values)

    @property
    def process_noise_trace(self) -> np.float64:
        return np.float64(np.trace(self.Q))

    @property
    def process_noise_log_det(self) -> np.float64:
        sign, logdet = np.linalg.slogdet(self.Q)
        if sign <= 0:
            return np.float64(-np.inf)
        return np.float64(logdet)

    @property
    def process_noise_condition_number(self) -> np.float64:
        eigs = np.linalg.eigvalsh(self.Q)
        if eigs.min() <= 0:
            return np.float64(np.inf)
        return np.float64(eigs.max() / eigs.min())

    @property
    def measurement_noise_trace(self) -> np.float64:
        return np.float64(np.trace(self.R))

    @property
    def measurement_noise_log_det(self) -> np.float64:
        sign, logdet = np.linalg.slogdet(self.R)
        if sign <= 0:
            return np.float64(-np.inf)
        return np.float64(logdet)

    @property
    def measurement_noise_condition_number(self) -> np.float64:
        eigs = np.linalg.eigvalsh(self.R)
        if eigs.min() <= 0:
            return np.float64(np.inf)
        return np.float64(eigs.max() / eigs.min())

    @property
    def steady_state_state_covariance(self) -> NDArray[np.float64]:
        self._assert_schur_stable()
        return solve_discrete_lyapunov(self.A, self.Q)

    @property
    def steady_state_signal_to_noise_ratio(self) -> np.float64:
        signal = np.trace(self.C @ self.steady_state_state_covariance @ self.C.T)
        noise = np.trace(self.R)
        if noise == 0:
            return np.float64(np.inf)
        return np.float64(signal / noise)

    @property
    def steady_state_filter_covariance(self) -> NDArray[np.float64]:
        return solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)

    @property
    def steady_state_kalman_gain(self) -> NDArray[np.float64]:
        P = self.steady_state_filter_covariance
        S = self.C @ P @ self.C.T + self.R
        return self.A @ P @ self.C.T @ np.linalg.inv(S)

    @property
    def filter_closed_loop_spectral_radius(self) -> np.float64:
        K = self.steady_state_kalman_gain
        A_f = self.A - K @ self.C
        return np.float64(np.max(np.abs(np.linalg.eigvals(A_f))))

    @property
    def filter_covariance_trace(self) -> np.float64:
        return np.float64(np.trace(self.steady_state_filter_covariance))

    @property
    def filter_covariance_log_det(self) -> np.float64:
        sign, logdet = np.linalg.slogdet(self.steady_state_filter_covariance)
        if sign <= 0:
            return np.float64(-np.inf)
        return np.float64(logdet)

    @property
    def innovation_covariance_trace(self) -> np.float64:
        return np.float64(
            np.trace(self.C @ self.steady_state_filter_covariance @ self.C.T + self.R)
        )

    SCALAR_PROPERTY_NAMES: tuple[str, ...] = (
        "spectral_radius_A",
        "spectral_gap_A",
        "dominant_mode_time_constant",
        "dominant_mode_damping_ratio",
        "dominant_mode_oscillation_frequency",
        "henrici_non_normality_index",
        "controllability_min_eigenvalue",
        "controllability_trace",
        "controllability_log_det",
        "controllability_condition_number",
        "controllability_effective_rank",
        "observability_min_eigenvalue",
        "observability_trace",
        "observability_log_det",
        "observability_condition_number",
        "observability_effective_rank",
        "hankel_norm",
        "hankel_min_singular_value",
        "hankel_condition_number",
        "hankel_effective_rank",
        "process_noise_trace",
        "process_noise_log_det",
        "process_noise_condition_number",
        "measurement_noise_trace",
        "measurement_noise_log_det",
        "measurement_noise_condition_number",
        "steady_state_signal_to_noise_ratio",
        "filter_closed_loop_spectral_radius",
        "filter_covariance_trace",
        "filter_covariance_log_det",
        "innovation_covariance_trace",
    )

    @property
    def scalar_properties(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.SCALAR_PROPERTY_NAMES}

    @property
    def scalar_property_vector(self) -> NDArray[np.float64]:
        return np.fromiter(
            (getattr(self, name) for name in self.SCALAR_PROPERTY_NAMES),
            dtype=np.float64,
            count=len(self.SCALAR_PROPERTY_NAMES),
        )

    def __repr__(self) -> str:
        return (
            f"LinearGaussianModel("
            f"state_dim={self.state_dim}, "
            f"input_dim={self.input_dim}, "
            f"obs_dim={self.obs_dim})"
        )
